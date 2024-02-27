
import numpy as np
import cython as pyx

if pyx.compiled:
    from cython.cimports.libc.math import round as round

if pyx.compiled:
    print('Running through Cython!')
else:
    print('WARNING: Not Compiled.')

LOWER = (1 - 0.9999) / 2
UPPER = 1 - LOWER

__ADD__:pyx.int = 0
__MUL__:pyx.int = 1
__MAX__:pyx.int = 2
__MIN__:pyx.int = 3
__SUB__:pyx.int = 4
__POW__:pyx.int = 5

DEFAULTS = {
    'maxBins':32,
    'compress':True,
    'tolerance':1e-3
}





@pyx.cclass                     
class RandomVariable():
    _lower: pyx.int[:]
    _upper: pyx.int[:]
    _known: pyx.double[:]
    _vague: pyx.double[:]
    _count: pyx.double[:]

    lower:   np.ndarray
    upper:   np.ndarray
    known:   np.ndarray
    vague: np.ndarray
    count:    np.ndarray

    maxBins: pyx.int
    nActive: pyx.int

    iMinUpper: pyx.int
    iMaxLower: pyx.int
    costLowerMax: pyx.double
    costUpperMin: pyx.double
    
    def __init__(self, maxBins=None, data=None, counts=None):
        if maxBins is None:
            maxBins = DEFAULTS['maxBins']

        self.lower = np.zeros(maxBins, dtype=np.intc)
        self.upper = np.zeros(maxBins, dtype=np.intc)
        self.count = np.zeros(maxBins, dtype=np.float64)
        self.known = np.zeros(maxBins, dtype=np.float64)
        self.vague = np.zeros(maxBins, dtype=np.float64)

        self._lower = self.lower
        self._upper = self.upper
        self._count = self.count
        self._known = self.known
        self._vague = self.vague

        self.nActive = 0
        self.maxBins = maxBins

        if (data is not None) and (counts is not None):
            self.fit(data, counts)

    def activeBins(self):
        return self.nActive

    def getCountArray(self):
        return self.count

    def getLowerArray(self):
        return self.lower

    def getUpperArray(self):
        return self.upper

    def setLower(self, lower):
        for i in range(lower.shape[0]):
            self.lower[i] = lower[i]

    def setUpper(self, upper):
        for i in range(upper.shape[0]):
            self.upper[i] = upper[i]

    def setCount(self, count):
        for i in range(count.shape[0]):
            self.count[i] = count[i]

    @pyx.ccall
    def setKnown(self, known):
        for i in range(known.shape[0]):
            self.known[i] = known[i]

    @pyx.ccall
    def lowerBound(self) -> pyx.int:
        return self._lower[0]

    @pyx.ccall
    def upperBound(self) -> pyx.int:
        # Because the upper bound is not inclusive we need to subtract one.
        return self._upper[self.nActive - 1] - 1

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)     
    def _countSum(self) -> pyx.double:
        i: pyx.int
        som: pyx.double = 0
        for i in range(self.nActive):
            som += self._count[i]

        return som

    def cdf(self, k):
        out = 0
        for i in range(self.nActive):
            if self.upper[i] <= k:
                # K is higher than the current bin.
                out += self.count[i] / self.count.sum()

            elif self.lower[i] <= k < self.upper[i]:
                #  k is within the bin.
                w = self.upper[i] - self.lower[i]
                p = self.count[i] / (self.count.sum() * w)
                out += p*(k - self.lower[i])

            else:
                # k is under the bin.
                break

        return out

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)        
    def pmf(self, k: pyx.int) -> pyx.double:
        i: pyx.int
        countSum: pyx.double = 0

        for i in range(self.nActive):
            countSum += self._count[i]
        
        
        for i in range(self.nActive):
            if self._lower[i] <= k < self._upper[i]:
                # Each point in the bin has the same probability.
                w: pyx.int    = self._upper[i] - self._lower[i]
                p: pyx.double = self._count[i] / (countSum * w)
                return p
        else:
            return 0

    def _assertConnected(self):
        for i in range(self.nActive):
            print(self.lower[i], self.upper[i])
        
        for i in range(self.nActive - 1):

            assert self.upper[i] == self.lower[i + 1]        

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)    
    def _sortBins(self):
        """
        We implement our own sorting algorithm here for two reasons. We need
        to sort more than one array using the same indexes and reindexing
        arrays create a copy in numpy. This breaks memoryviews in calling
        functions who still refer to the original copy. Secondly, because
        we know that the array will be partially sorted using insertion
        sort will theoretically be faster.
        """
        lowerKey: pyx.int
        upperKey: pyx.int
        knownKey: pyx.double
        vagueKey: pyx.double
        countKey: pyx.double
        i: pyx.int
        j: pyx.int

        for i in range(1, self.nActive):
            lowerKey = self._lower[i]
            upperKey = self._upper[i]
            knownKey = self._known[i]
            vagueKey = self._vague[i]
            countKey = self._count[i]

            j = i - 1

            while j >= 0 and lowerKey < self._lower[j]:
                self._lower[j+1] = self._lower[j]
                self._upper[j+1] = self._upper[j]
                self._known[j+1] = self._known[j]
                self._vague[j+1] = self._vague[j]
                self._count[j+1] = self._count[j]
                j -= 1

            self._lower[j+1] = lowerKey
            self._upper[j+1] = upperKey
            self._known[j+1] = knownKey
            self._vague[j+1] = vagueKey
            self._count[j+1] = countKey



    @pyx.cfunc
    def _addPhaseOne(self, k:pyx.int, weight:pyx.double):
        """
        Adds a point when we haven't already filled all the different histograms.
        """
        lower:   pyx.int[:] = self.lower
        upper:   pyx.int[:] = self.upper
        known:   pyx.double[:] = self.known
        #vague: pyx.double[:] = self.vague
        count: pyx.double[:] = self.count #

        
        i: pyx.int = self._findBin(k)
        if i >= 0:
            count[i]  += weight
            known[i] += weight
        else:
            i = self.nActive

            # self.lower[i] = k
            # self.upper[i] = k + 1
            lower[i] = k
            upper[i] = k + 1            
            count[i] = weight
            known[i] = weight

            self.nActive += 1

            if self.nActive == self.maxBins:
                self._sortBins()

                # During phase 1 bins are created "unconnected". We need to connect 
                # them before we continue. This could break down when we don't have
                # all the bins filled.

                for i in range(self.nActive - 1):
                    # self.upper[i] = self.lower[i + 1]
                    # print('->',self.upper[i], self.lower[i+1])
                    
                    upper[i] = lower[i + 1]

    @pyx.cfunc
    def _fillBinGaps(self):
        """
        From time to time it happens that the bins are not connected because we didn't
        fill maxBins but the data points are not close to each other. In these cases
        we need to make sure that the bins are connected.
        """
        self._sortBins()

        for i in range(1, self.nActive):
            if self.lower[i] != self.upper[i-1]:
                self.lower[i] = self.upper[i-1]

        



    @pyx.cfunc
    def _findBin(self,k:pyx.int) -> pyx.int:

        nActive: pyx.int    = self.nActive
        lower:   pyx.int[:] = self.lower
        upper:   pyx.int[:] = self.upper

        for i in range(nActive):
            if lower[i] <= k < upper[i]:
                return i
        else:
            return -1

    @pyx.cfunc
    @pyx.boundscheck(False) 
    def _addPhaseTwo(self, k: pyx.int, weight: pyx.double):
        """
        """
        lower:   pyx.int[:] = self.lower
        upper:   pyx.int[:] = self.upper
        count:   pyx.double[:] = self.count
        known:   pyx.double[:] = self.known
        
        if k < lower[0]:
            # We need to stretch the lower bin to accomodate the new point.
            lower[0]  = k
            count[0]  += weight
            known[0] += weight

        elif k >= upper[-1]:
            # We need to stretch the upper bin to accomodate the new point.
            upper[-1]  = k + 1
            count[-1]  += weight
            known[-1] += weight
        else:
            i: pyx.int = self._findBin(k)
            
            count[i]  += weight
            known[i] += weight  

        
    @pyx.cfunc
    @pyx.boundscheck(False)    
    def _merge(self, iMin: pyx.int):
        # Stretches the iMin bin to encompass the iMin+1 bin as well. This clears
        # up the iMin+1 bin to use for splitting.

        lower: pyx.int[:]   = self.lower
        upper: pyx.int[:]   = self.upper
        count: pyx.double[:] = self.count
        known: pyx.double[:] = self.known
        vague: pyx.double[:] = self.vague        

        upper[iMin] = upper[iMin + 1]
        count[iMin] = count[iMin]    + count[iMin + 1]
        known[iMin] = known[iMin]   + known[iMin + 1]
        vague[iMin] = vague[iMin] + vague[iMin + 1]

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)
    def _split(self, iMax: pyx.int, m1: pyx.int):
        # Splits iMax into two bins and stores the one in m1.

        lower: pyx.int[:]   = self.lower
        upper: pyx.int[:]   = self.upper
        count: pyx.double[:] = self.count
        known: pyx.double[:] = self.known
        vague: pyx.double[:] = self.vague
        
        l:pyx.int = lower[iMax]
        u:pyx.int = upper[iMax]

        w: pyx.int = u - l

        fa: pyx.double = count[iMax]
        fk: pyx.double = known[iMax]
        fu: pyx.double = vague[iMax]

        m2: pyx.int = iMax

        lower[m1] = l
        upper[m1] = l + pyx.cast(pyx.int, round(w/2))
        known[m1] = 0
        vague[m1] = fk + fu
        count[m1] = fa / 2


        lower[m2] = l + pyx.cast(pyx.int, round(w/2))
        upper[m2] = u
        known[m2] = 0
        vague[m2] = fk + fu        
        count[m2] = fa / 2

    @pyx.cfunc
    @pyx.boundscheck(False)
    def _findMinMax(self):
        """
        Performs the following operations but does so in a single pass.
        
        costLower = (upper - lower) * (known - vague)
        costUpper = (upper - lower) * (known + vague)

        adjCostUpper = costUpper[:-1] + costUpper[1:]

        iMaxLower: pyx.int = np.argmax(costLower)
        iMinUpper: pyx.int = np.argmin(adjCostUpper)

        """
        lower: pyx.int[:]      = self.lower
        upper: pyx.int[:]      = self.upper
        known: pyx.double[:]   = self.known
        vague: pyx.double[:] = self.vague

        iMinUpper: pyx.int = -1
        iMaxLower: pyx.int = -1

        costLowerMax: pyx.double = -9e9
        costUpperMin: pyx.double = 9e9


        k: pyx.int


        for k in range(self.nActive):
            costLower: pyx.double = (upper[k] - lower[k]) * (known[k] - vague[k])

            if costLower > costLowerMax:
                iMaxLower = k
                costLowerMax = costLower


            if k < self.nActive - 2:
                # We use -2 here because it is zero based indexing and we want to
                # stop one before the last value.
                adjCostUpper: pyx.double
                adjCostUpper  = (upper[k] - lower[k]) * (known[k] + vague[k])
                adjCostUpper += (upper[k+1] - lower[k+1]) * (known[k+1] + vague[k+1])

                if adjCostUpper < costUpperMin:
                    iMinUpper = k
                    costUpperMin = adjCostUpper

        self.iMinUpper    = iMinUpper
        self.iMaxLower    = iMaxLower
        self.costLowerMax = costLowerMax
        self.costUpperMin = costUpperMin

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)      
    def _frequencyCount(self, data: pyx.int[:], counts: pyx.double[:]):
        # f: pyx.double[:] = np.zeros(self.maxBins) 
        nD: pyx.int = len(data)
        d: pyx.int
        i: pyx.int

        for i in range(self.nActive):
            self._count[i] = 0

        for d in range(nD):
            for i in range(self.nActive):
                if self._lower[i] <= data[d] < self._upper[i]:
                    self._count[i] += counts[d]
                    break
            else:
                self._count[-1] += counts[d]

        return self._count

    @pyx.ccall
    def fit(self, data, counts):
        print('TEST!!')
        idx    = np.argsort(data)
        data   = data[idx]
        counts = counts[idx]

        minD: pyx.int = data[ 0]
        maxD: pyx.int = data[-1]

        if maxD - minD > self.maxBins:
            self._presetBins(minD, maxD)

            ii:pyx.int = 0
            while True:
                self._frequencyCount(data, counts)

                cost = (self.upper - self.lower) * self.count
                cost_n = cost[:-1] + cost[1:]

                iMax: pyx.int = np.argmax(cost)
                iMin: pyx.int = np.argmin(cost_n)

                if cost[iMax] <= 2*cost_n[iMin]:
                    break

                self._merge(iMin)
                self._split(iMax, iMin + 1)
                self._sortBins()
                ii += 1
                if ii >= 100:
                    break

            self.setKnown(self.count)
            self.nActive = self.maxBins
        else:
            # self._presetBins(minD, maxD)
            for di, ci in zip(data, counts):
                self.add(di, ci)

            self._fillBinGaps()

    @pyx.cfunc
    def _presetBins(self, minK: pyx.int, maxK: pyx.int):
        """
        This presets some of the bins so that we have a nice spreadout of bins for convolution,
        otherwise the bins are all clustered around the left point.
        """
        print('_presetBins()')
        nK: pyx.int = maxK - minK + 1
        if nK > self.maxBins:
            # Only do something if the number of points will exceed the number of
            # bins that we are going to use.
            bins: pyx.int[:] = np.zeros(self.maxBins + 1, dtype=np.intc)
            bins[0] = minK

            stp: pyx.int
            for stp in range(self.maxBins):
                rK_remain: pyx.int = maxK - bins[stp]
                nB_remain: pyx.int = self.maxBins - stp
                dK_remain: pyx.int = int(np.ceil(rK_remain / nB_remain))

                bins[stp+1] = bins[stp] + dK_remain

            bins[-1] = bins[-1] + 1

            self.setLower(bins[:-1].copy())
            self.setUpper(bins[1:].copy())

            # nBins = bins[:-1].shape[0]
            self.nActive = bins[:-1].shape[0]

        
    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)    
    def add(self, k:pyx.int, weight:pyx.double=1):
        k = pyx.cast(pyx.int, round(k))

        if self.nActive < self.maxBins:
            self._addPhaseOne(k, weight=weight)
        else:
            self._addPhaseTwo(k, weight=weight)

            self._findMinMax()

            overlap: pyx.int = (self.iMinUpper == self.iMaxLower) or (self.iMinUpper + 1 == self.iMaxLower)

            wMax: pyx.int = self._upper[self.iMaxLower] - self._lower[self.iMaxLower]

            if (self.costLowerMax > 2*self.costUpperMin) and (not overlap) and (wMax > 1):
                self._merge(self.iMinUpper)
                self._split(self.iMaxLower, self.iMinUpper + 1)
                self._sortBins()


    @pyx.ccall
    # @pyx.boundscheck(False)
    # @pyx.initializedcheck(False)                
    def toArray(self):

        lowerK: pyx.int = self.lowerBound()
        upperK: pyx.int = self.upperBound()

        nK: pyx.int = upperK - lowerK + 1

        # print(f'{upperK} - {lowerK} = {nK}')
        
        outW = np.zeros(nK)
        outK = np.zeros(nK, dtype=np.intc)

        _outW: pyx.double[:] = outW
        _outK: pyx.int[:]    = outK

        k: pyx.int

        for k in range(lowerK, upperK+1):
            _outK[k - lowerK] = k
            _outW[k - lowerK] = self.pmf(k)

        # som = 0
        # k = self.lowerBound()
        # while som < UPPER:
        #     pk = self.pmf(k)
        #     som += pk
        #     outW.append(pk)
        #     outK.append(k)            
        #     k   += 1


        return outK, outW

    
    @pyx.cfunc
    def _applyFunc(self, iS: pyx.int, iO:pyx.int, func:pyx.int) -> pyx.int:
        __ADD__:pyx.int = 0
        __MUL__:pyx.int = 1
        __MAX__:pyx.int = 2
        __MIN__:pyx.int = 3
        __SUB__:pyx.int = 4
        __POW__:pyx.int = 5        
        iF: pyx.int = 0
        if func == __ADD__:
            iF = iS + iO
        elif func == __MUL__:
            iF = iO*iS
        elif func == __MAX__:
            iF = iO if iO > iS else iS
        elif func == __MIN__:
            iF = iO  if iO < iS else iS
        elif func == __SUB__:
            iF = iS - iO
        elif func == __POW__:
            iF = iS**iO

        return iF

    @pyx.ccall
    @pyx.boundscheck(False)
    def __conv__(self, other, func: pyx.int):
        
        final = RandomVariable()

        _kS, _pS = self.toArray()
        _kO, _pO = other.toArray()

        kS: pyx.int[:] = _kS
        pS: pyx.double[:] = _pS

        kO: pyx.int[:] = _kO
        pO: pyx.double[:] = _pO

        nS: pyx.int = len(kS)
        nO: pyx.int = len(kO)

        s: pyx.int
        o: pyx.int
        # print('==================================================')

        data   = np.zeros(nS*nO, dtype=np.intc)
        counts = np.zeros(nS*nO)
        _data: pyx.int[:]      = data
        _counts: pyx.double[:] = counts
        i: pyx.int = 0

        minK = self._applyFunc(kS[0], kO[0], func)
        maxK = self._applyFunc(kS[-1], kO[-1], func)
        
        final._presetBins(minK,maxK)

        
        for s in range(nS):
            for o in range(nO):
                pF: pyx.double = pS[s] * pO[o]
                kF: pyx.int = self._applyFunc(kS[s], kO[o], func)

                # print(f'P[{kS[s]} + {kO[o]} = {kF}] => {pF} ')

                _data[i]   = kF
                _counts[i] = pF
                i += 1

                if pF > 0:
                    pass
                    # final.add(kF, pF) 

        final.fit(data, counts)
        print(final.lower)
        print(final.upper)
        print(final.known)
        print(final.count)
        # final.compress()

        return final

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)     
    def compress(self):
        """
        Peforms lossy compression on the RandomVariable histogram by compressing
        the first and last bin.
        """
        if self._count[0]/self._countSum() <= 1e-3:
            w: pyx.int    = self._upper[0] - self._lower[0]
            f: pyx.double = self._count[0]
            k: pyx.double = self._known[0]

            self._lower[0] = self._upper[0] - 1
            self._count[0] = f / w
            self._known[0] = k / w

        if self._count[-1]/self._countSum() <= 1e-3:
            w: pyx.int    = self._upper[-1] - self._lower[-1]
            f: pyx.double = self._count[-1]
            k: pyx.double = self._known[-1]

            self._lower[-1] = self._upper[-1] - 1
            self._count[-1] = f / w
            self._known[-1] = k / w            
            
            
            
        

    def __add__(self, other):
        return self.__conv__(other, __ADD__)

    def __sub__(self, other):
        return self.__conv__(other, __SUB__)

    def __mul__(self, other):
        return self.__conv__(other, __MUL__)

    def __max__(self, other):
        return self.__conv__(other, __MAX__)
      
    def __min__(self, other):
        return self.__conv__(other, __MIN__)      

    def __pow__(self, other):
        return self.__conv__(other, __POW__)        







        

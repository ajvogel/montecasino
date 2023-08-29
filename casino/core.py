import numpy as np
import cython as pyx

from cython.cimports.libc.math import round as round

if pyx.compiled:
    print('Running through Cython!')
else:
    print('Running in Python')

LOWER = 0.001
UPPER = 1 - LOWER

__ADD__:pyx.int = 0
__MUL__:pyx.int = 1
__MAX__:pyx.int = 2
__MIN__:pyx.int = 3
__SUB__:pyx.int = 4
__POW__:pyx.int = 5

DEFAULTS = {
    'maxBins':16
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
    
    def __init__(self, maxBins=None):
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

    def activeBins(self):
        return self.nActive

    def getCountArray(self):
        return self.count

    def lowerBound(self):
        return self.lower[0]

    def upperBound(self):
        return self.upper[-1]

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


    def pmf(self, k):
        for i in range(len(self.upper)):
            if self.lower[i] <= k < self.upper[i]:
                # Each point in the bin has the same probability.
                w = self.upper[i] - self.lower[i]
                p = self.count[i] / (self.count.sum() * w)
                return p
        else:
            return -1

    def _assertConnected(self):
        for i in range(self.nActive):
            print(self.lower[i], self.upper[i])
        
        for i in range(self.nActive - 1):

            assert self.upper[i] == self.lower[i + 1]        

    def _sortBins(self):
        """
        We implement our own sorting algorithm here for two reasons. We need
        to sort more than one array using the same indexes and reindexing
        arrays create a copy in numpy. This breaks memoryviews in calling
        functions who still refer to the original copy. Secondly, because
        we know that the array will be partially sorted using insertion
        sort will theoretically be faster.
        """
        
        # idx: pyx.long[:] = np.argsort(self.lower)

        # self.lower   = self.lower[idx]
        # self.upper   = self.upper[idx]
        # self.known   = self.known[idx]
        # self.vague = self.vague[idx]
        # self.count    = self.count[idx]

        lowerKey: pyx.int
        upperKey: pyx.int
        knownKey: pyx.double
        vagueKey: pyx.double
        countKey: pyx.double
        i: pyx.int
        j: pyx.int

        for i in range(1, self.nActive):
            lowerKey   = self.lower[i]
            upperKey   = self.upper[i]
            knownKey   = self.known[i]
            vagueKey = self.vague[i]
            countKey    = self.count[i]

            j = i - 1

            while j >= 0 and lowerKey < self.lower[j]:
                self.lower[j+1]   = self.lower[j]
                self.upper[j+1]   = self.upper[j]
                self.known[j+1]   = self.known[j]
                self.vague[j+1] = self.vague[j]
                self.count[j+1]    = self.count[j]
                j -= 1

            self.lower[j+1]   = lowerKey
            self.upper[j+1]   = upperKey
            self.known[j+1]   = knownKey
            self.vague[j+1] = vagueKey
            self.count[j+1]    = countKey

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
            count[i]  = weight
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
        count:    pyx.double[:] = self.count
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
        count[iMin]    = count[iMin]    + count[iMin + 1]
        known[iMin]   = known[iMin]   + known[iMin + 1]
        vague[iMin] = vague[iMin] + vague[iMin + 1]

    @pyx.cfunc
    @pyx.boundscheck(False)
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


        self.iMinUpper = iMinUpper
        self.iMaxLower = iMaxLower
        self.costLowerMax = costLowerMax
        self.costUpperMin = costUpperMin



                    

    @pyx.ccall
    @pyx.boundscheck(False)
    def add(self, k:pyx.int, weight:pyx.double=1):
        k = pyx.cast(pyx.int, round(k))

        lower = self.lower
        upper = self.upper
        known = self.known      
        vague = self.vague

        # iMinUpper: pyx.int
        # iMaxLower: pyx.int
        # costUpperMin: pyx.double
        # costLowerMax: pyx.double
        
        if self.nActive < self.maxBins:
            self._addPhaseOne(k, weight=weight)
        else:
            self._addPhaseTwo(k, weight=weight)

            # self._findMinMax()
            # iMinUpper, costUpperMin, iMaxLower, costLowerMax = self._findMinMax() 
            costLower = (upper - lower) * (known - vague)
            costUpper = (upper - lower) * (known + vague)

            adjCostUpper = costUpper[:-1] + costUpper[1:]

            iMaxLower: pyx.int = np.argmax(costLower)
            iMinUpper: pyx.int = np.argmin(adjCostUpper)

            self.iMaxLower = iMaxLower
            self.iMinUpper = iMinUpper

            self.costLowerMax = costLower[self.iMaxLower]
            self.costUpperMin = costUpper[self.iMinUpper]

            overlap: pyx.int = (self.iMinUpper == self.iMaxLower) or (self.iMinUpper + 1 == self.iMaxLower)

            wMax: pyx.int = upper[self.iMaxLower] - lower[self.iMaxLower]

            if (self.costLowerMax > 2*self.costUpperMin) and (not overlap) and (wMax > 1):
                self._merge(self.iMinUpper)
                self._split(self.iMaxLower, self.iMinUpper + 1)
                self._sortBins()


    def toArray(self):
        outW = []
        outK = []

        som = 0
        k = self.lowerBound()
        while som < UPPER:
            pk = self.pmf(k)
            som += pk
            outW.append(pk)
            outK.append(k)            
            k   += 1


        return np.array(outK), np.array(outW)

    
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

        kS: pyx.long[:] = _kS
        pS: pyx.double[:] = _pS

        kO: pyx.long[:] = _kO
        pO: pyx.double[:] = _pO

        nS: pyx.int = len(kS)
        nO: pyx.int = len(kO)

        s: pyx.int
        o: pyx.int

        for s in range(nS):
            for o in range(nO):
                pF: pyx.double = pS[s] * pO[o]
                kF: pyx.int = self._applyFunc(kS[s], kO[o], func)

                if pF > 0:
                    final.add(kF, pF)

        return final

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







        

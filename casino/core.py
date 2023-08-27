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
    # _lower:   pyx.double[:]
    # _upper:   pyx.double[:]
    # _known:   pyx.double[:]
    # _unknown: pyx.double[:]
    # _freq:    pyx.double[:]

    lower:   np.ndarray
    upper:   np.ndarray
    known:   np.ndarray
    unknown: np.ndarray
    freq:    np.ndarray

    maxBins: pyx.int
    nActive: pyx.int
    
    def __init__(self, maxBins=None):
        if maxBins is None:
            maxBins = DEFAULTS['maxBins']

        self.lower   = np.zeros(maxBins, dtype=np.intc)
        self.upper   = np.zeros(maxBins, dtype=np.intc)
        self.freq    = np.zeros(maxBins, dtype=np.float64)
        self.known   = np.zeros(maxBins, dtype=np.float64)
        self.unknown = np.zeros(maxBins, dtype=np.float64)

        # self._lower   = self.lower
        # self._upper   = self.upper
        # self._freq    = self.freq
        # self._known   = self.known
        # self._unknown = self.unknown

        self.nActive = 0
        self.maxBins = maxBins

    def lowerBound(self):
        return self.lower[0]

    def upperBound(self):
        return self.upper[-1]

    def cdf(self, k):
        out = 0
        for i in range(self.nActive):
            if self.upper[i] <= k:
                # K is higher than the current bin.
                out += self.freq[i] / self.freq.sum()

            elif self.lower[i] <= k < self.upper[i]:
                #  k is within the bin.
                w = self.upper[i] - self.lower[i]
                p = self.freq[i] / (self.freq.sum() * w)
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
                p = self.freq[i] / (self.freq.sum() * w)
                return p
        else:
            return -1

    def _assertConnected(self):
        for i in range(self.nActive - 1):
            assert self.upper[i] == self.lower[i + 1]        

    @pyx.cfunc
    def _sortBins(self):
        
        idx: pyx.long[:] = np.argsort(self.lower)

        self.lower   = self.lower[idx]
        self.upper   = self.upper[idx]
        self.known   = self.known[idx]
        self.unknown = self.unknown[idx]

    @pyx.cfunc
    def _addPhaseOne(self, k:pyx.int, weight:pyx.double):
        """
        Adds a point when we haven't already filled all the different histograms.
        """
        i: pyx.int = self._findBin(k)
        if i >= 0:
            self.freq[i]  += weight
            self.known[i] += weight
        else:
            i = self.nActive

            self.lower[i] = k
            self.upper[i] = k + 1
            self.freq[i]  = weight
            self.known[i] = weight

            self.nActive += 1

            if self.nActive == self.maxBins:
                self._sortBins()

                # During phase 1 bins are created "unconnected". We need to connect 
                # them before we continue. This could break down when we don't have
                # all the bins filled.

                for i in range(self.nActive - 1):
                    self.upper[i] = self.lower[i + 1]


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
        freq:    pyx.double[:] = self.freq
        known:   pyx.double[:] = self.known
        
        if k < lower[0]:
            # We need to stretch the lower bin to accomodate the new point.
            lower[0]  = k
            freq[0]  += weight
            known[0] += weight

        elif k >= upper[-1]:
            # We need to stretch the upper bin to accomodate the new point.
            upper[-1]  = k + 1
            freq[-1]  += weight
            known[-1] += weight
        else:
            i: pyx.int = self._findBin(k)
            
            freq[i]  += weight
            known[i] += weight  

        
    @pyx.cfunc
    @pyx.boundscheck(False)    
    def _merge(self, iMin: pyx.int):
        # Stretches the iMin bin to encompass the iMin+1 bin as well. This clears
        # up the iMin+1 bin to use for splitting.

        lower: pyx.int[:]   = self.lower
        upper: pyx.int[:]   = self.upper
        freq: pyx.double[:] = self.freq
        known: pyx.double[:] = self.known
        unknown: pyx.double[:] = self.unknown        

        upper[iMin] = upper[iMin + 1]
        freq[iMin]    = freq[iMin]    + freq[iMin + 1]
        known[iMin]   = known[iMin]   + known[iMin + 1]
        unknown[iMin] = unknown[iMin] + unknown[iMin + 1]

    @pyx.cfunc
    @pyx.boundscheck(False)
    def _split(self, iMax: pyx.int, m1: pyx.int):
        # Splits iMax into two bins and stores the one in m1.

        lower: pyx.int[:]   = self.lower
        upper: pyx.int[:]   = self.upper
        freq: pyx.double[:] = self.freq
        known: pyx.double[:] = self.known
        unknown: pyx.double[:] = self.unknown
        
        l:pyx.int = lower[iMax]
        u:pyx.int = upper[iMax]

        w: pyx.int = u - l

        fa: pyx.double = freq[iMax]
        fk: pyx.double = known[iMax]
        fu: pyx.double = unknown[iMax]

        m2: pyx.int = iMax

        lower[m1] = l
        upper[m1] = l + pyx.cast(pyx.int, round(w/2))
        known[m1] = 0
        unknown[m1] = fk + fu
        freq[m1] = fa / 2


        lower[m2] = l + pyx.cast(pyx.int, round(w/2))
        upper[m2] = u
        known[m2] = 0
        unknown[m2] = fk + fu        
        freq[m2] = fa / 2        

    @pyx.ccall
    def add(self, k:pyx.int, weight:pyx.double=1):
        k = pyx.cast(pyx.int, round(k))

        lower = self.lower
        upper = self.upper
        known = self.known
        unknown = self.unknown
        
        if self.nActive < self.maxBins:
            self._addPhaseOne(k, weight=weight)
        else:
            self._addPhaseTwo(k, weight=weight)

            costLower = (upper - lower) * (known - unknown)
            costUpper = (upper - lower) * (known + unknown)

            adjCostUpper = costUpper[:-1] + costUpper[1:]

            iMaxLower: pyx.int = np.argmax(costLower)
            iMinUpper: pyx.int = np.argmin(adjCostUpper)

            overlap = (iMinUpper == iMaxLower) or (iMinUpper + 1 == iMaxLower)

            wMax: pyx.int = upper[iMaxLower] - lower[iMaxLower]

            if (costLower[iMaxLower] > 2*costUpper[iMinUpper]) and (not overlap) and (wMax > 1):
                self._merge(iMinUpper)
                self._split(iMaxLower, iMinUpper + 1)
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

    def __conv__(self, other, func):
        
        final = RandomVariable()

        kS, pS = self.toArray()
        kO, pO = other.toArray()

        nS: pyx.int = len(kS)
        nO: pyx.int = len(kO)

        s: pyx.int
        o: pyx.int

        for s in range(nS):
            for o in range(nO):
                pF = pS[s] * pO[o]
                kF = self._applyFunc(kS[s], kO[o], func)

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







        

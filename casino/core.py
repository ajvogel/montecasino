import numpy as np

import cython as pyx

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



class RandomVariable():
    def __init__(self, maxBins=None):
        if maxBins is None:
            maxBins = DEFAULTS['maxBins']

        self.lower   = np.zeros(maxBins)
        self.upper   = np.zeros(maxBins)
        self.freq    = np.zeros(maxBins)
        self.known   = np.zeros(maxBins)
        self.unknown = np.zeros(maxBins)

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

    def _sortBins(self):
        idx = np.argsort(self.lower)

        self.lower   = self.lower[idx]
        self.upper   = self.upper[idx]
        self.known   = self.known[idx]
        self.unknown = self.unknown[idx]

    def _addPhaseOne(self, k, weight):
        """
        Adds a point when we haven't already filled all the different histograms.
        """
        i = self._findBin(k)
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

    def _findBin(self,k:pyx.int) -> pyx.int:

        nActive: pyx.int       = self.nActive
        lower:   pyx.double[:] = self.lower
        upper:   pyx.double[:] = self.upper

        for i in range(nActive):
            if lower[i] <= k < upper[i]:
                return i
        else:
            return -1

    def _addPhaseTwo(self, k: pyx.int, weight: pyx.double):
        """
        """
        lower:   pyx.double[:] = self.lower
        upper:   pyx.double[:] = self.upper
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

        

    def _merge(self, iMin):
        # Stretches the iMin bin to encompass the iMin+1 bin as well. This clears
        # up the iMin+1 bin to use for splitting.

        self.upper[iMin] = self.upper[iMin + 1]

        self.freq[iMin]    = self.freq[iMin]    + self.freq[iMin + 1]
        self.known[iMin]   = self.known[iMin]   + self.known[iMin + 1]
        self.unknown[iMin] = self.unknown[iMin] + self.unknown[iMin + 1]

    def _split(self, iMax, m1):
        # Splits iMax into two bins and stores the one in m1.
        l = self.lower[iMax]
        u = self.upper[iMax]

        w = u - l

        fa = self.freq[iMax]
        fk = self.known[iMax]
        fu = self.unknown[iMax]

        m2 = iMax

        self.lower[m1] = l
        self.upper[m1] = l + round(w/2)
        self.known[m1] = 0
        self.unknown[m1] = fk + fu
        self.freq[m1] = fa / 2


        self.lower[m2] = l + round(w/2)
        self.upper[m2] = u
        self.known[m2] = 0
        self.unknown[m2] = fk + fu        
        self.freq[m2] = fa / 2        


    def add(self, k, weight=1):
        k = round(k)

        if self.nActive < self.maxBins:
            self._addPhaseOne(k, weight=weight)
        else:
            self._addPhaseTwo(k, weight=weight)

            costLower = (self.upper - self.lower) * (self.known - self.unknown)
            costUpper = (self.upper - self.lower) * (self.known + self.unknown)

            adjCostUpper = costUpper[:-1] + costUpper[1:]

            iMaxLower = np.argmax(costLower)
            iMinUpper = np.argmin(adjCostUpper)

            overlap = (iMinUpper == iMaxLower) or (iMinUpper + 1 == iMaxLower)

            wMax = self.upper[iMaxLower] - self.lower[iMaxLower]

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

        nS, nO = len(kS), len(kO)

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







        

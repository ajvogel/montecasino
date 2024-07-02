
from types import prepare_class

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





class RandomVariable():
    def __init__(self, maxBins=32):
        self.maxBins = maxBins
        self.nActive = 0

        self.bins = np.zeros(self.maxBins + 1)
        self.cnts = np.zeros(self.maxBins + 1)

    def _findLastLesserOrEqualIndex(self, point):
        idx = -1
        while True:
            if (self.bins[idx + 1] > point) or (idx + 1 == self.nActive):
                break
            else:
                idx += 1

        return idx

    def _shiftRightAndInsert(self, idx, point, count):

        for j in range(self.nActive - 1, idx, -1):
            self.bins[j+1] = self.bins[j]
            self.cnts[j+1] = self.cnts[j]

        self.bins[idx+1] = point
        self.cnts[idx+1] = count
        self.nActive += 1

    def _findMinimumDifference(self):
        minK    = -1
        minDiff = 9e9
        # We don't want to merge the first or last bin because we want to maintain the
        # tails. It also solves the problem where we try and sample a point that is before
        # the first centroid.
        for k in range(1, self.nActive - 2):
            dB = self.bins[k+1] - self.bins[k]
            if dB < minDiff:
                minDiff = dB
                minK    = k

        return minK

    def _shiftLeftAndOverride(self, idx):
        for j in range(idx, self.nActive-1):
            self.bins[j] = self.bins[j+1]
            self.cnts[j] = self.cnts[j+1]

        self.bins[self.nActive-1] = 0
        self.cnts[self.nActive-1] = 0

        self.nActive -= 1

    def fit(self, x):
        for xx in x:
            self.add(xx)

    def add(self, point, count=1):

        idx = self._findLastLesserOrEqualIndex(point)
                
        if (idx >= 0) and self.bins[idx] == point:
            self.cnts[idx] += count
        else:
            self._shiftRightAndInsert(idx, point, count)

        if self.nActive > self.maxBins:
            k = self._findMinimumDifference()

            sumC = self.cnts[k+1] + self.cnts[k]

            self.bins[k] = (self.bins[k]*self.cnts[k] + self.bins[k+1]*self.cnts[k+1])
            self.bins[k] = self.bins[k] / sumC
            self.cnts[k] = sumC

            self._shiftLeftAndOverride(k+1)

    def cdf(self, k):
        som = 0

        m = self.cnts
        b = self.bins

        if k < b[0]:
            return 0
        elif b[0] <= k <= b[self.nActive - 1]:
            
            for i in range(self.nActive):
                if b[i] <= k < b[i+1]:
                    mb   = m[i] + (m[i+1] - m[i])/(b[i+1] - b[i])*(k - b[i])
                    som += (m[i] + mb)/2 * (k - b[i])/(b[i+1] - b[i])
                    som += m[i]/2
                    break
                else:
                    som += m[i]

            return som / m.sum()

        else:
            return 1

    def pmf(self, k):

        if self.nActive < self.maxBins:
            i = self._findLastLesserOrEqualIndex(k)
            return self.cnts[i]

        
        if (k < self.bins[0]) or (k > self.bins[self.nActive - 1]):
            return 0

        i = self._findLastLesserOrEqualIndex(k)
        p = self.bins
        w = self.cnts

        N = np.floor(p[i+1]) - np.ceil(p[i]) + 1
        m = (w[i+1] - w[i])/(p[i+1] - p[i])

        y0 = m*(np.ceil(p[i]) - p[i]) + w[i]

        S = (N/2)*(2*y0 + m*(N-1))
        W = w.sum()

        fx = m*(k - p[i]) + w[i]


        return (fx / (2*W*S))*(w[i] + w[i+1])

    def lower(self):
        return int(self.bins[0])

    def upper(self):
        return int(self.bins[self.nActive - 1])

    def sample(self, size=1):
        pass

    def quantile(self, p):
        pass

    def compress(self):
        pass

    # --- Convolution related functions....
    

    def _applyFunc(self, iS: pyx.int, iO:pyx.int, func:pyx.int) -> pyx.int:

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
        
    def __conv__(self, other, func: pyx.int):
        
        final = RandomVariable()

        for s in range(self.lower(), self.upper() + 1):
            for o in range(other.lower(), other.upper() + 1):
                pF = self.pmf(s) * other.pmf(o)
                kF = self._applyFunc(s, o, func)

                if pF > 0:
                    final.add(kF, pF)

        final.compress()
        return final

                




        

        # _kS, _pS = self.toArray()
        # _kO, _pO = other.toArray()

        # # print(f'    {_kS[0]} -> {_kS[-1]} ({len(_kS)}); {_kO[0]} -> {_kO[-1]} ({len(_kO)})')

        # kS: pyx.int[:] = _kS
        # pS: pyx.double[:] = _pS

        # kO: pyx.int[:] = _kO
        # pO: pyx.double[:] = _pO

        # nS: pyx.int = len(kS)
        # nO: pyx.int = len(kO)

        # s: pyx.int
        # o: pyx.int
        # # print('==================================================')

        # data   = np.zeros(nS*nO, dtype=np.intc)
        # counts = np.zeros(nS*nO)
        # _data: pyx.int[:]      = data
        # _counts: pyx.double[:] = counts
        # i: pyx.int = 0

        # minK = self._applyFunc(kS[0], kO[0], func)
        # maxK = self._applyFunc(kS[-1], kO[-1], func)
        
        # final._presetBins(minK,maxK)

        
        # for s in range(nS):
        #     for o in range(nO):
        #         pF: pyx.double = pS[s] * pO[o]
        #         kF: pyx.int = self._applyFunc(kS[s], kO[o], func)

        #         # print(f'P[{kS[s]} + {kO[o]} = {kF}] => {pF} ')

        #         _data[i]   = kF
        #         _counts[i] = pF
        #         i += 1

        #         # if pF > 0:
        #         #     pass
        #         #     final.add(kF, pF) 

        # final.fit(data, counts)
        # # print(final.lower)
        # # print(final.upper)
        # # print(final.known)
        # # print(final.count)
        # final.compress()

        # return final



        
        # # Find the last index equal or smaller.
        # idx = -1
        # while True:
        #     if (self.bins[idx + 1] > point) or (idx + 1 == self.nActive):
        #         break
        #     else:
        #         idx += 1
                
        # if self.bins[idx] == point:
        #     self.cnts[idx] += count
            
            
        # # Find the index one before the place we should add the new number.
        # if self.nActive == 0:
        #     i = -1
        # else:
        #     for i in range(self.nActive):
        #         if point > self.bins[i]:
        #             break

        # if (i >= 0) and (self.bins[i] == point):
        #     # We already have a bin with this number. Adding this to the current count.
        #     self.cnts[i] += count
        # else:
        #     # We must move all the points to the right to make space for the new point
        #     # we want to add.
        #     for j in range(self.nActive, i, -1):
        #         self.bins[j + 1] = self.bins[j]
        #         self.cnts[j + 1] = self.cnts[j]

        #     # Insert the new bin in.
        #     self.bins[i+1] = point
        #     self.cnts[i+1] = count

        #     self.nActive += 1

        # if self.nActive > self.maxBins:
        #     # Find minimum k index.
        #     minK    = -1
        #     minDiff = 9e9
        #     for k in range(self.nActive):
        #         if (self.bins[k+1] - self.bins[k]) < minDiff:
        #             minDiff = self.bins[k+1] - self.bins[k]
        #             minK    = k

        #     # Replace bins k and k+1 with.
        #     sumCnts = self.cnts[minK] + self.cnts[minK + 1]
        #     self.bins[minK] = (self.bins[minK]*self.cnts[minK] + self.bins[minK+1]*self.cnts[minK+1]) / sumCnts
        #     self.cnts[minK] = sumCnts

        #     for j in range(minK + 1, self.nActive):
        #         self.bins[j] = self.bins[j+1]
        #         self.cnts[j] = self.cnts[j+1]

        #     self.nActive -= 1
            

#===============================================================================



# @pyx.cclass                     
# class RandomVariable():
#     _bins:   pyx.int[:]
#     _cnts: pyx.double[:]

#     bins: np.ndarray
#     cnts: np.ndarray

#     maxBins: pyx.int
#     nActive: pyx.int

#     def __init__(self, maxBins=None, data=None, counts=None):
#         if maxBins is None:
#             maxBins = DEFAULTS['maxBins']

#         self.bins = np.zeros(maxBins, dtype=np.intc)
#         self.cnts = np.zeros(maxBins, dtype=np.float64)

#         self.nActive = 0
#         self.maxBins = maxBins

#         if (data is not None) and (counts is not None):
#             self.fit(data, counts)

#     # ---[ Assertions ]-----------------------------------------------------------------------------
#     def _assertCompiled(self):
#         assert pyx.compiled

#     def _assertConnected(self):
#         i: pyx.int
        
#         for i in range(self.nActive - 1):
#             assert self._upper[i] == self._lower[i + 1]

#     # ---[ Utility Methods ]------------------------------------------------------------------------
#     @pyx.ccall
#     @pyx.boundscheck(False)
#     @pyx.initializedcheck(False) 
#     def add(self, point, count:pyx.double = 1):

#         i: pyx.int

#         for i in range(self.nActive):
#             if point == self._bins[i]:
#                 self._cnts[i] += count
#         else:
#             if self.nActive == self.maxBins:
#                 j = self._mergeBins()
#             else:
#                 j = self.nActive
#                 self.nActive += 1

#             self._bins[j] = point
#             self._cnts[j] = count

#             self._sortBins()

#     @pyx.cfunc
#     @pyx.boundscheck(False)
#     @pyx.initializedcheck(False)            
#     def _mergeBins(self) -> pyx.int:
#         i: pyx.int
#         minDff: pyx.int
#         minI: pyx.int
        
#         minDff = self._bins[1] - self._bins[0]
#         minI = 0 

#         for i in range(self.nActive - 1):
#             if self._bins[i+1] - self._bins[i] < minDff:
#                 minDff = self._bins[i+1] - self._bins[i]
#                 minI = i

#         i = minI
#         self._bins[i]  = (self._bins[i]*self._cnts[i] + self._bins[i+1]*self._cnts[i+1])
#         self._bins[i] /= (self._cnts[i] + self._cnts[i+1])
#         self._cnts[i]  = self._cnts[i] + self._cnts[i+1]

#         return i + 1
                

#     @pyx.cfunc
#     @pyx.boundscheck(False)
#     @pyx.initializedcheck(False)    
#     def _sortBins(self):
#         """
#         We implement our own sorting algorithm here for two reasons. We need to sort more than one
#         array using the same indexes and reindexing arrays create a copy in numpy. This breaks
#         memoryviews in calling functions who still refer to the original copy. Secondly, because we
#         know that the array will be partially sorted using insertion sort will theoretically be faster.
#         """
#         i: pyx.int
#         j: pyx.int

#         for i in range(1, self.nActive):
#             binKey = self._bins[i]
#             cntKey = self._cnts[i]

#             j = i - 1

#             while j >= 0 and binKey < self._bins[j]:
#                 self._bins[j+1] = self._bins[j]
#                 self._cnts[j+1] = self._cnts[j]

#                 j -= 1

#             self._bins[j+1] = binKey
#             self._cnts[j+1] = cntKey



                    

                
    # ---[ Getters / Setters ]----------------------------------------------------------------------
            
    # def activeBins(self):
    #     return self.nActive

    # def getCountArray(self):
    #     return self.count

    # def getLowerArray(self):
    #     return self.lower

    # def getUpperArray(self):
    #     return self.upper

    # def setLower(self, lower):
    #     for i in range(lower.shape[0]):
    #         self.lower[i] = lower[i]

    # def setUpper(self, upper):
    #     for i in range(upper.shape[0]):
    #         self.upper[i] = upper[i]

    # def setCount(self, count):
    #     for i in range(count.shape[0]):
    #         self.count[i] = count[i]

    # @pyx.ccall
    # def setKnown(self, known):
    #     for i in range(known.shape[0]):
    #         self.known[i] = known[i]

    # @pyx.ccall
    # def lowerBound(self) -> pyx.int:
    #     return self._lower[0]

    # @pyx.ccall
    # def upperBound(self) -> pyx.int:
    #     # Because the upper bound is not inclusive we need to subtract one.
    #     return self._upper[self.nActive - 1] - 1



    # def cdf(self, k):
    #     pass
        # out = 0
        # for i in range(self.nActive):
        #     if self.upper[i] <= k:
        #         # K is higher than the current bin.
        #         out += self.count[i] / self.count.sum()

        #     elif self.lower[i] <= k < self.upper[i]:
        #         #  k is within the bin.
        #         w = self.upper[i] - self.lower[i]
        #         p = self.count[i] / (self.count.sum() * w)
        #         out += p*(k - self.lower[i])

        #     else:
        #         # k is under the bin.
        #         break

        # return out

    # @pyx.ccall
    # @pyx.boundscheck(False)
    # @pyx.initializedcheck(False)        
    # def pmf(self, k: pyx.int) -> pyx.double:
    #     pass
        # i: pyx.int
        # countSum: pyx.double = 0

        # for i in range(self.nActive):
        #     countSum += self._count[i]
        
        
        # for i in range(self.nActive):
        #     if self._lower[i] <= k < self._upper[i]:
        #         # Each point in the bin has the same probability.
        #         w: pyx.int    = self._upper[i] - self._lower[i]
        #         p: pyx.double = self._count[i] / (countSum * w)
        #         return p
        # else:
        #     return 0











        



    # @pyx.cfunc
    # def _findBin(self,k:pyx.int) -> pyx.int:

    #     nActive: pyx.int    = self.nActive
    #     lower:   pyx.int[:] = self.lower
    #     upper:   pyx.int[:] = self.upper

    #     for i in range(nActive):
    #         if lower[i] <= k < upper[i]:
    #             return i
    #     else:
    #         return -1

    # @pyx.cfunc
    # @pyx.boundscheck(False) 
    # def _addPhaseTwo(self, k: pyx.int, weight: pyx.double):
    #     """
    #     """
    #     lower:   pyx.int[:] = self.lower
    #     upper:   pyx.int[:] = self.upper
    #     count:   pyx.double[:] = self.count
    #     known:   pyx.double[:] = self.known
        
    #     if k < lower[0]:
    #         # We need to stretch the lower bin to accomodate the new point.
    #         lower[0]  = k
    #         count[0]  += weight
    #         known[0] += weight

    #     elif k >= upper[-1]:
    #         # We need to stretch the upper bin to accomodate the new point.
    #         upper[-1]  = k + 1
    #         count[-1]  += weight
    #         known[-1] += weight
    #     else:
    #         i: pyx.int = self._findBin(k)
            
    #         count[i]  += weight
    #         known[i] += weight  

        
    # @pyx.cfunc
    # @pyx.boundscheck(False)    
    # def _merge(self, iMin: pyx.int):
    #     # Stretches the iMin bin to encompass the iMin+1 bin as well. This clears
    #     # up the iMin+1 bin to use for splitting.

    #     lower: pyx.int[:]   = self.lower
    #     upper: pyx.int[:]   = self.upper
    #     count: pyx.double[:] = self.count
    #     known: pyx.double[:] = self.known
    #     vague: pyx.double[:] = self.vague        

    #     upper[iMin] = upper[iMin + 1]
    #     count[iMin] = count[iMin]    + count[iMin + 1]
    #     known[iMin] = known[iMin]   + known[iMin + 1]
    #     vague[iMin] = vague[iMin] + vague[iMin + 1]

    # @pyx.cfunc
    # @pyx.boundscheck(False)
    # @pyx.initializedcheck(False)
    # def _split(self, iMax: pyx.int, m1: pyx.int):
    #     # Splits iMax into two bins and stores the one in m1.

    #     lower: pyx.int[:]   = self.lower
    #     upper: pyx.int[:]   = self.upper
    #     count: pyx.double[:] = self.count
    #     known: pyx.double[:] = self.known
    #     vague: pyx.double[:] = self.vague
        
    #     l:pyx.int = lower[iMax]
    #     u:pyx.int = upper[iMax]

    #     w: pyx.int = u - l

    #     fa: pyx.double = count[iMax]
    #     fk: pyx.double = known[iMax]
    #     fu: pyx.double = vague[iMax]

    #     m2: pyx.int = iMax

    #     lower[m1] = l
    #     upper[m1] = l + pyx.cast(pyx.int, round(w/2))
    #     known[m1] = 0
    #     vague[m1] = fk + fu
    #     count[m1] = fa / 2


    #     lower[m2] = l + pyx.cast(pyx.int, round(w/2))
    #     upper[m2] = u
    #     known[m2] = 0
    #     vague[m2] = fk + fu        
    #     count[m2] = fa / 2





    # @pyx.ccall
    # def fit(self, data, counts):
    #     pass
        


    
    # @pyx.cfunc
    # def _applyFunc(self, iS: pyx.int, iO:pyx.int, func:pyx.int) -> pyx.int:
    #     __ADD__:pyx.int = 0
    #     __MUL__:pyx.int = 1
    #     __MAX__:pyx.int = 2
    #     __MIN__:pyx.int = 3
    #     __SUB__:pyx.int = 4
    #     __POW__:pyx.int = 5        
    #     iF: pyx.int = 0

    #     if func == __ADD__:
    #         iF = iS + iO
    #     elif func == __MUL__:
    #         iF = iO*iS
    #     elif func == __MAX__:
    #         iF = iO if iO > iS else iS
    #     elif func == __MIN__:
    #         iF = iO  if iO < iS else iS
    #     elif func == __SUB__:
    #         iF = iS - iO
    #     elif func == __POW__:
    #         iF = iS**iO

    #     return iF

    # @pyx.ccall
    # @pyx.boundscheck(False)
    # def __conv__(self, other, func: pyx.int):
        
    #     final = RandomVariable()

    #     _kS, _pS = self.toArray()
    #     _kO, _pO = other.toArray()

    #     # print(f'    {_kS[0]} -> {_kS[-1]} ({len(_kS)}); {_kO[0]} -> {_kO[-1]} ({len(_kO)})')

    #     kS: pyx.int[:] = _kS
    #     pS: pyx.double[:] = _pS

    #     kO: pyx.int[:] = _kO
    #     pO: pyx.double[:] = _pO

    #     nS: pyx.int = len(kS)
    #     nO: pyx.int = len(kO)

    #     s: pyx.int
    #     o: pyx.int
    #     # print('==================================================')

    #     data   = np.zeros(nS*nO, dtype=np.intc)
    #     counts = np.zeros(nS*nO)
    #     _data: pyx.int[:]      = data
    #     _counts: pyx.double[:] = counts
    #     i: pyx.int = 0

    #     minK = self._applyFunc(kS[0], kO[0], func)
    #     maxK = self._applyFunc(kS[-1], kO[-1], func)
        
    #     final._presetBins(minK,maxK)

        
    #     for s in range(nS):
    #         for o in range(nO):
    #             pF: pyx.double = pS[s] * pO[o]
    #             kF: pyx.int = self._applyFunc(kS[s], kO[o], func)

    #             # print(f'P[{kS[s]} + {kO[o]} = {kF}] => {pF} ')

    #             _data[i]   = kF
    #             _counts[i] = pF
    #             i += 1

    #             # if pF > 0:
    #             #     pass
    #             #     final.add(kF, pF) 

    #     final.fit(data, counts)
    #     # print(final.lower)
    #     # print(final.upper)
    #     # print(final.known)
    #     # print(final.count)
    #     final.compress()

    #     return final

    # @pyx.ccall
    # @pyx.boundscheck(False)
    # @pyx.initializedcheck(False)     
    # def compress(self):
    #     """
    #     Peforms lossy compression on the RandomVariable histogram by compressing
    #     the first and last bin.
    #     """
    #     pass
            
            
            
        

    # def __add__(self, other):
    #     return self.__conv__(other, __ADD__)

    # def __sub__(self, other):
    #     return self.__conv__(other, __SUB__)

    # def __mul__(self, other):
    #     return self.__conv__(other, __MUL__)

    # def __max__(self, other):
    #     return self.__conv__(other, __MAX__)
      
    # def __min__(self, other):
    #     return self.__conv__(other, __MIN__)      

    # def __pow__(self, other):
    #     return self.__conv__(other, __POW__)        







        

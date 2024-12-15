
from types import prepare_class

import numpy as np
import cython as pyx

if pyx.compiled:
    from cython.cimports.libc.math import round as round
    from cython.cimports.libc.math import ceil as ceil
    from cython.cimports.libc.math import floor as floor
else:
    ceil = np.ceil
    floor = np.floor
    
    

if pyx.compiled:
    print('Running through Cython!')
else:
    print('WARNING: Not Compiled.')







#---[ VirtualMachine ]----------------------------------------------------------

PASS    = 0
PUSH    = 1
ADD     = 2
MUL     = 3
POW     = 4
RANDINT = 5

class VirtualMachine():
    def __init__(self, codes, operands) -> None:
        self.codes    = codes
        self.operands = operands
        self.stack    = np.zeros(100)
        self.stackCount = 0

    def pushStack(self, value):
        self.stack[self.stackCount] = value
        self.stackCount += 1

    def popStack(self):
        assert self.stackCount > 0
        self.stackCount -= 1
        return self.stack[self.stackCount]

    def _add(self):
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 + x2)

    def _mul(self):
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 * x2)        
        
    def _pow(self):
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 ** x2)

    def _randInt(self):
        l = self.popStack()
        h = self.popStack()
        self.pushStack(np.random.randint(l, h))
        
    def run(self):

        N     = self.codes.shape(0)
        i     = 0

        while i < N:
            opCode = self.codes[i]

            if   opCode == PASS:
                pass
            elif opCode == PUSH:    self.pushStack(self.operands[i])
            elif opCode == ADD:     self._add()
            elif opCode == MUL:     self._mul()
            elif opCode == POW:     self._pow()
            elif opCode == RANDINT: self._randInt()

            i += 1

        return self.popStack()
                
                
            
















    

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
    _bins: pyx.double[:]
    _cnts: pyx.double[:]

    bins: np.ndarray
    cnts: np.ndarray

    maxBins: pyx.int
    nActive: pyx.int
    
    def __init__(self, maxBins=32):
        self.maxBins = maxBins
        self.nActive = 0

        self.bins = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.cnts = np.zeros(self.maxBins + 1, dtype=np.float64)

        self._bins = self.bins
        self._cnts = self.cnts

    def setBins(self, bins):

        self.bins = bins

        # If we don't set self._bins as well it will still refer to the previous
        # array and not the new one. In essence self._bins points to the memory
        # block and not the name self.bins
        
        self._bins = self.bins

    def getBins(self):
        return self.bins

    def setWeights(self, cnts):
        self.cnts = cnts
        self._cnts = self.cnts

    def getWeights(self):
        return self.cnts

    def setActiveBinCount(self, nActive):
        self.nActive = nActive

    def getActiveBinCount(self):
        return self.nActive

    def _assertCompiled(self):
        assert pyx.compiled    

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)    
    @pyx.initializedcheck(False)        
    def _findLastLesserOrEqualIndex(self, point:pyx.double) -> pyx.int:

        idx:pyx.int = -1
        while True:
            if (self._bins[idx + 1] > point) or (idx + 1 == self.nActive):
                break
            else:
                idx += 1

        return idx

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)    
    @pyx.initializedcheck(False)            
    def _shiftRightAndInsert(self, idx:pyx.int, point:pyx.double, count:pyx.double) -> pyx.void:
        j: pyx.int

        for j in range(self.nActive - 1, idx, -1):
            self._bins[j+1] = self._bins[j]
            self._cnts[j+1] = self._cnts[j]

        self._bins[idx+1] = point
        self._cnts[idx+1] = count
        self.nActive += 1

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)    
    @pyx.initializedcheck(False)
    def _findMinimumDifference(self) -> pyx.int:
        k: pyx.int
        dB: pyx.double
        minK: pyx.int
        minDiff: pyx.double
        
        minK    = -1
        minDiff = 9e9
        # We don't want to merge the first or last bin because we want to maintain the
        # tails. It also solves the problem where we try and sample a point that is before
        # the first centroid.
        for k in range(1, self.nActive - 2):
            dB = self._bins[k+1] - self._bins[k]
            if dB < minDiff:
                minDiff = dB
                minK    = k

        return minK

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)    
    @pyx.initializedcheck(False)                
    def _shiftLeftAndOverride(self, idx: pyx.int) -> pyx.void:
        j: pyx.int
        for j in range(idx, self.nActive-1):
            self._bins[j] = self._bins[j+1]
            self._cnts[j] = self._cnts[j+1]

        self._bins[self.nActive-1] = 0
        self._cnts[self.nActive-1] = 0

        self.nActive -= 1

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)    
    @pyx.initializedcheck(False)
    def _sumWeights(self) -> pyx.double:
        som: pyx.double = 0
        i: pyx.int

        for i in range(self.nActive):
            som = som + self._cnts[i]

        return som

    def fit(self, x):
        for xx in x:
            self.add(xx)

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)        
    @pyx.initializedcheck(False)                            
    def _add(self, point:pyx.double, count:pyx.double) -> pyx.void:

        idx:pyx.int = self._findLastLesserOrEqualIndex(point)
                
        if (idx >= 0) and self._bins[idx] == point:
            self._cnts[idx] += count
        else:
            self._shiftRightAndInsert(idx, point, count)

        if self.nActive > self.maxBins:
            k:pyx.int = self._findMinimumDifference()

            sumC:pyx.double = self._cnts[k+1] + self._cnts[k]

            self._bins[k] = (self._bins[k]*self._cnts[k] + self._bins[k+1]*self._cnts[k+1])
            self._bins[k] = self._bins[k] / sumC
            self._cnts[k] = sumC

            self._shiftLeftAndOverride(k+1)

    def centroids(self):
        return self.bins[:-1]

    def weights(self):
        return self.cnts[:-1]

    def add(self, point, count=1):
        self._add(point, count)

    def cdf(self, k):
        """

        Ted Dunning, Computing Extremely Accurate Quantiles Usings t-Digests
        """
        som = 0
        c = self.bins
        m = self.cnts

        if k < self.lower():
            return 0.
        elif k >= self.upper():
            return 1.
        else:
            for i in range(self.nActive):
                if c[i] <= k < c[i+1]:
                    # We use the approach of Dunning here to improve interpolation when
                    # we have single weighted points.
                    if (m[i] > 1) & (m[i+1] > 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som + m[i]/2
                        yi_n = yi + (m[i+1] + m[i]) / 2

                    elif (m[i] == 1) & (m[i+1] > 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som
                        yi_n = yi + (m[i+1]) / 2

                    elif (m[i] > 1) & (m[i+1] == 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som + m[i]/2
                        yi_n = yi + (m[i]) / 2
                    else:
                        yi   = som
                        yi_n = yi
                        
                    g    = (yi_n - yi) / (c[i+1] - c[i])
                    yk   = g*(k - c[i]) + yi

                    return yk / self._sumWeights()

                else:
                    som += m[i]

            print(k)
            print(self.getBins())
            print(self.getWeights())

    # def cdf(self, k):
    #     som = 0

    #     m = self.cnts
    #     b = self.bins

    #     if k < b[0]:
    #         return 0
    #     elif b[0] <= k <= b[self.nActive - 1]:
            
    #         for i in range(self.nActive):
    #             if b[i] <= k < b[i+1]:
    #                 mb   = m[i] + (m[i+1] - m[i])/(b[i+1] - b[i])*(k - b[i])
    #                 som += (m[i] + mb)/2 * (k - b[i])/(b[i+1] - b[i])
    #                 som += m[i]/2
    #                 break
    #             else:
    #                 som += m[i]

    #         return som / m.sum()

    #     else:
    #         return 1

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)    
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)         
    def pmf(self, kk: pyx.int) -> pyx.double:
        return self.cdf(kk + 0.5) - self.cdf(kk - 0.5)

        
        # i: pyx.int
        # p: pyx.double[:]
        # w: pyx.double[:]
        # k: pyx.double
        # N: pyx.double
        # m: pyx.double
        # S: pyx.double
        # W: pyx.double
        # fx: pyx.double

        
        # k = pyx.cast(pyx.double, kk)

        # if self.nActive < self.maxBins:
        #     i = self._findLastLesserOrEqualIndex(k)
        #     return self._cnts[i]

        
        # if (k < self._bins[0]) or (k > self._bins[self.nActive - 1]):
        #     return 0

        # i = self._findLastLesserOrEqualIndex(k)
        # p = self._bins
        # w = self._cnts

        # N = floor(p[i+1]) - ceil(p[i]) + 1
        # m = (w[i+1] - w[i])/(p[i+1] - p[i])

        # y0 = m*(ceil(p[i]) - p[i]) + w[i]

        # S = (N/2)*(2*y0 + m*(N-1))
        # # W = self.cnts.sum()
        # W = self._sumWeights()

        # fx = m*(k - p[i]) + w[i]


        # return (fx / (2*W*S))*(w[i] + w[i+1])

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)    
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)        
    def lower(self) -> pyx.int:
        return int(self._bins[0])

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)    
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)        
    def upper(self) -> pyx.int:
        return int(self._bins[self.nActive - 1])

    def sample(self, size=1):
        p = np.random.rand()
        return int(round(self.quantile(p)))


    def quantile(self, p):
        # Using Regula Falsi
        a = self.lower()
        b = self.upper()

        fa = 0 - p
        fb = 1 - p

        iter = 0

        while ((b - a) > 0.1) and (iter < 100):
            c = (fb*a - fa*b) / (fb - fa)
            fc = self.cdf(c) - p

            if fc < 0:
                a  = c
                fa = fc
            elif fc > 0:
                b  = c
                fb = fc
            else:
                return c

            iter += 1

        return (b+a)/2


        
    
    # def quantile(self, p):
    #     c  = self._bins
    #     m  = self._cnts

    #     S  = self._sumWeights()
    #     Sp = S*p
    #     s = 0

    #     for i in range(-1, self.nActive):
    #         if i == -1:
    #             ci   = c[i+1] - 0.1
    #             ci_n = c[i+1]
    #             mi   = 0
    #             mi_n = m[i+1]
    #         elif i == self.nActive - 1:
    #             ci   = c[i]
    #             ci_n = c[i] + 0.1
    #             mi   = m[i]
    #             mi_n = 0               
    #         else:
    #             ci   = c[i]
    #             ci_n = c[i+1]
    #             mi   = m[i]
    #             mi_n = m[i+1]
            
    #         w = (mi + mi_n) / 2

    #         if s <= Sp <= (s + w):
    #             g = (mi_n - mi)/(ci_n - ci)
    #             A = Sp - s

    #             cp = (-mi + (mi**2 + 2*g*A)**0.5)/g + ci
    #             return cp

    #         else:
    #             s += w
            

    def compress(self):
        pass

    # --- Convolution related functions....
    
    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.initializedcheck(False)
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

    @pyx.ccall
    def __conv__(self, other, func: pyx.int):
        s: pyx.int
        o: pyx.int
        i: pyx.int
        pF: pyx.double
        kF: pyx.double
        final = RandomVariable()


        # for i in range(100000):
        #     s = self.sample()
        #     o = other.sample()
        #     kF = self._applyFunc(s, o, func)
        #     final._add(kF, 1)
            

        for s in range(self.lower(), self.upper() + 1):
            for o in range(other.lower(), other.upper() + 1):
                pF = self.pmf(s) * other.pmf(o)
                kF = self._applyFunc(s, o, func)

                if pF > 0:
                    final._add(kF, pF)

        # final.compress()
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







        

import numpy as np
import cython as pyx

if pyx.compiled:
    from cython.cimports.libc.math import round as c_round
    from cython.cimports.libc.math import ceil as ceil
    from cython.cimports.libc.math import floor as floor
    from cython.cimports.libc.time import time as c_time
    from cython.cimports.libc.stdlib import rand as c_rand
    from cython.cimports.libc.stdlib import srand as c_srand
    from cython.cimports.libc.stdlib import RAND_MAX as c_RAND_MAX
    from cython.cimports.random import normal
else:
    ceil = np.ceil
    floor = np.floor



if pyx.compiled:
    print('Running through Cython!')
else:
    print('WARNING: Not Compiled.')


c_srand(c_time(pyx.NULL))

#======================================[ Random Variable ]=========================================


#==================================================================================================

#---[ Probability Distributions ]-----------------------------------------------

# Rand
@pyx.cfunc
def _rand() -> pyx.double:
    out:pyx.double = pyx.cast(pyx.double, c_rand()) / pyx.cast(pyx.double, c_RAND_MAX)
    return out

# RandInt
@pyx.cfunc
def _randint(l: pyx.double, h: pyx.double) -> pyx.double:
    return c_round((h - l) * _rand() + l)






# NormalDistribution


# Pert Distribution


# Negative Binomial




#---[ VirtualMachine ]----------------------------------------------------------

PASS:pyx.int    = 0
PUSH:pyx.int    = 1
ADD:pyx.int    = 2
MUL:pyx.int     = 3
POW:pyx.int     = 4
RANDINT:pyx.int = 5

@pyx.cclass
class VirtualMachine():
    _codes: pyx.long[:]
    _operands: pyx.double[:]
    _stack: pyx.double[:]

    codes: np.ndarray
    operands: np.ndarray
    stack: np.ndarray
    stackCount: pyx.int
    def __init__(self, codes, operands) -> None:
        self.codes    = codes
        self.operands = operands
        self.stack    = np.zeros(100)
        self.stackCount = 0

    # Init the memory view.fdd
        self._codes = self.codes
        self._operands = self.operands
        self._stack   = self.stack

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def pushStack(self, value: pyx.double):
        self._stack[self.stackCount] = value
        self.stackCount += 1

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def popStack(self) -> pyx.float:
        assert self.stackCount > 0
        self.stackCount -= 1
        return self._stack[self.stackCount]


    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _add(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 + x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _mul(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 * x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _pow(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 ** x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randInt(self) -> pyx.void:
        h = self.popStack()
        l = self.popStack()

        # x: pyx.double = c_round((h - l)*_rand() + l)

        self.pushStack(pyx.cast(pyx.double, _randint(l,h)))

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def compute(self) -> pyx.float:
        PASS:pyx.int    = 0
        PUSH:pyx.int    = 1
        ADD:pyx.int    = 2
        MUL:pyx.int     = 3
        POW:pyx.int     = 4
        RANDINT:pyx.int = 5
        N:pyx.int = self._codes.shape[0]
        i:pyx.int = 0
        opCode: pyx.long

        while i < N:
            opCode = self._codes[i]

            if   opCode == PASS:
                pass
            elif opCode == PUSH:    self.pushStack(self._operands[i])
            elif opCode == ADD:     self._add()
            elif opCode == MUL:     self._mul()
            elif opCode == POW:     self._pow()
            elif opCode == RANDINT: self._randInt()

            i += 1

        return self.popStack()

    def sample(self, samples:pyx.int=10000, maxBins:pyx.int=32):
        rv = RandomVariable(maxBins=maxBins)
        i: pyx.int
        for i in range(samples):
            x:pyx.float = self.compute()
            rv._add(x, 1)

        return rv

    def run(self):
        return self.compute()









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



    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def pmf(self, kk: pyx.int) -> pyx.double:
        return self.cdf(kk + 0.5) - self.cdf(kk - 0.5)


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

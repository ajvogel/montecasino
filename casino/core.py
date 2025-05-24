import numpy as np
import cython as pyx

import random

if pyx.compiled:
    from cython.cimports.libc.math import round as c_round
    from cython.cimports.libc.math import ceil as c_ceil
    from cython.cimports.libc.math import ceil as ceil
    from cython.cimports.libc.math import floor as c_floor
    from cython.cimports.libc.time import time as c_time
    from cython.cimports.libc.math import log as c_log
    from cython.cimports.libc.math import sqrt as c_sqrt
    from cython.cimports.libc.stdlib import rand as c_rand
    from cython.cimports.libc.stdlib import srand as c_srand
    from cython.cimports.libc.stdlib import RAND_MAX as c_RAND_MAX
    # from cython.cimports.random import normal
else:
    ceil = np.ceil
    floor = np.floor



if pyx.compiled:
    print('Running through Cython!')
else:
    print('WARNING: Not Compiled.')


c_srand(c_time(pyx.NULL))

#======================================[ Random Variable ]=========================================






@pyx.cclass
class Digest():
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
        p: pyx.double = _rand()
        #p = np.random.rand()
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
    l2: pyx.double  = l - 1
    out: pyx.double = c_ceil((h - l2) * _rand() + l2)

    if out < l:
        out = l

    return out

    #return c_round((h - l) * _rand() + l)

@pyx.cfunc
def _randnorm(mu: pyx.double, stdev: pyx.double) -> pyx.double:
    """
    https://en.wikipedia.org/wiki/Marsaglia_polar_method
    """
    while True:
        x:pyx.double = 2*_rand() - 1
        y:pyx.double = 2*_rand() - 1

        s: pyx.double = x**2 + y**2

        if s < 1:
            break

    z: pyx.double = x * c_sqrt( -2*c_log(s) / s)
    return z * stdev + mu








#======================================[ Virtual Machine ]=========================================
# pyx.declare creates c constants.
_PASS = pyx.declare(pyx.int, 0)
_PUSH = pyx.declare(pyx.int, 1)

_DROP  = pyx.declare(pyx.int, 2)
_STORE = pyx.declare(pyx.int, 3)
_LOAD  = pyx.declare(pyx.int, 4)

# Onetary Ops
_NEG = pyx.declare(pyx.int, 10)
_ABS = pyx.declare(pyx.int, 11)

# Binary Ops
_ADD = pyx.declare(pyx.int, 20)
_MUL = pyx.declare(pyx.int, 21)
_POW = pyx.declare(pyx.int, 22)
_DIV = pyx.declare(pyx.int, 23)
_SUB = pyx.declare(pyx.int, 24)
_MOD = pyx.declare(pyx.int, 25)
_FLOORDIV = pyx.declare(pyx.int, 26)

_BINOPMAX = pyx.declare(pyx.int, 50)

# Summation Thingies...
#OP_SUM_START:pyx.int = 51
_SUM_START = pyx.declare(pyx.int, 51)
_SUM_END   = pyx.declare(pyx.int, 52)

# Statistical Ops
_RANDINT = pyx.declare(pyx.int, 100)
_RANDNORM = pyx.declare(pyx.int, 101)
_RAND_QUANTILES = pyx.declare(pyx.int, 102)
_ARRAY_SUM = pyx.declare(pyx.int, 103)

@pyx.cclass
class VirtualMachine():

    # Core Op Codes


    _codes: pyx.double[:]
    _operands: pyx.double[:]
    _stack: pyx.double[:]
    _variables: pyx.double[:]
    _pointers: pyx.long[:]
    _iterators: pyx.double[:]

    codes: np.ndarray
    operands: np.ndarray
    stack: np.ndarray
    variables: np.ndarray
    stackCount: pyx.int
    counter: pyx.int
    pointerCount: pyx.int
    pointers: np.ndarray
    iterCount: pyx.int
    iterators: np.ndarray
    def __init__(self, codes, operands) -> None:
        self.codes    = codes
        self.operands = operands
        self.stack    = np.zeros(100)
        self.variables = np.zeros(26)
        self.pointers = np.zeros(16, dtype=np.int_)
        self.iterators = np.zeros(16)
        self.stackCount = 0
        self.pointerCount = 0
        self.iterCount = 0
        self.counter = 0


        # Init the memory view.fdd
        self._codes = self.codes
        self._operands = self.operands
        self._stack   = self.stack
        self._variables = self.variables
        self._pointers = self.pointers
        self._iterators = self.iterators

    @pyx.ccall
    def reset(self):
        self.stackCount   = 0
        self.iterCount    = 0
        self.pointerCount = 0
        self.counter      = 0

        self.stack    = np.zeros(100)
        self.variables = np.zeros(26)
        self.pointers = np.zeros(16, dtype=np.int_)
        self.iterators = np.zeros(16)



    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushPointer(self, value: pyx.int) -> pyx.void:
        self._pointers[self.pointerCount] = value
        self.pointerCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popPointer(self) -> pyx.int:
        assert self.pointerCount > 0
        self.pointerCount -= 1
        return self._pointers[self.pointerCount]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushIterator(self, value: pyx.double) -> pyx.void:
        self._iterators[self.iterCount] = value
        self.iterCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popIterator(self) -> pyx.double:
        assert self.iterCount > 0
        self.iterCount -= 1
        return self._iterators[self.iterCount]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def peekIterator(self) -> pyx.double:
        """Returns the bottom pointer in the pointer stack without popping it from the stack"""
        return self._iterators[self.iterCount - 1]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def peekPointer(self) -> pyx.int:
        """Returns the bottom pointer in the pointer stack without popping it from the stack"""
        return self._pointers[self.pointerCount - 1]

    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushStack(self, value: pyx.double):
        self._stack[self.stackCount] = value
        self.stackCount += 1

    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popStack(self) -> pyx.float:
        assert self.stackCount > 0
        self.stackCount -= 1
        return self._stack[self.stackCount]

    @pyx.cfunc
    def _dropStack(self, cnt:pyx.int = 1):
        i: pyx.int
        #print('Dropping ',cnt, ' values...')
        for i in range(cnt):
            #print('    Drop', i, '...')
            self.popStack()

    @pyx.cfunc
    def _store(self, varNumber: pyx.double) -> pyx.void:
        idx: pyx.int = pyx.cast(pyx.int, varNumber)
        varValue = self.popStack()
        self._variables[idx] = varValue

    @pyx.cfunc
    def _load(self, varNumber: pyx.double) -> pyx.void:
        idx: pyx.int = pyx.cast(pyx.int, varNumber)
        self.pushStack(self._variables[idx])

    @pyx.cfunc
    def _sumStart(self, loopNumber: pyx.double) -> pyx.void:
        idx: pyx.int = pyx.cast(pyx.int, loopNumber)
        nTerms = self.popStack()
        self.pushStack(0)
        self.pushIterator(nTerms)
        self.pushPointer(self.counter)

    @pyx.cfunc
    def _sumEnd(self, loopNumber: pyx.double) -> pyx.void:

        idx: pyx.int = pyx.cast(pyx.int, loopNumber)

        # First we add the running total to the answer.
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 + x2)
        self.pushIterator(self.popIterator() - 1)

        if self.peekIterator() > 0:
            self.counter = self.peekPointer()

        else:
            self.popPointer()
            self.popIterator()



    @pyx.cfunc
    def _binop(self, opCode: pyx.double) -> pyx.void:
        # When things are removed from the stack we pop them from the bottom of the stack
        # in the reverse order in which they were pushed. So we pop x2 first before x1.
        # _ADD:pyx.int   = 20
        # _MUL:pyx.int   = 21
        # _POW:pyx.int   = 22
        # _DIV:pyx.int   = 23
        # _SUB:pyx.int   = 24
        # _MOD:pyx.int   = 25
        # _FLOORDIV: pyx.int = 26

        x2 = self.popStack()
        x1 = self.popStack()
        if opCode == _ADD:
            self.pushStack(x1 + x2)
        elif opCode == _MUL:
            self.pushStack(x1 * x2)
        elif opCode == _POW:
            self.pushStack(x1 ** x2)
        elif opCode == _DIV:
            self.pushStack(x1 / x2)
        elif opCode == _FLOORDIV:
            self.pushStack(x1 // x2)
        elif opCode == _MOD:
            self.pushStack(x1 % x2)
        elif opCode == _SUB:
            self.pushStack(x1 - x2)

    @pyx.cfunc
    def _randInt(self) -> pyx.void:
        h = self.popStack()
        l = self.popStack()

        # x: pyx.double = c_round((h - l)*_rand() + l)
        self.pushStack(_randint(l, h))
        #self.pushStack(pyx.cast(pyx.double, _randint(l,h)))
        #self.pushStack(random.randint(int(l),int(h)))

    @pyx.cfunc
    def _randNorm(self) -> pyx.void:
        std: pyx.double = self.popStack()
        mu: pyx.double    = self.popStack()

        self.pushStack(_randnorm(mu, std))

    @pyx.cfunc
    def _arraySum(self, nArray: pyx.double) -> pyx.void:
        som: pyx.double = 0.0
        i: pyx.int

        start: pyx.double = self.popStack()
        end:pyx.double    = self.popStack()

        for i in range(nArray):
            if start <= i < end:
                som += self.popStack()
        self.pushStack(som)

    @pyx.cfunc
    def _randQuantiles(self, nBins: pyx.double) -> pyx.void:
        """

        quants = 01 15 29 43 57 71 85 99
                 x0 x1 x2 x3 x4 x5 x6 x7

        """
        # There is one fewer intervals than there are actual points.
        dY: pyx.double = 1. / (nBins - 1)
        y_: pyx.double = _rand()
        #y_: pyx.double = 0.55


        i: pyx.double = c_floor(y_ / dY)

        #print(i)
        #print('Stack Count:', self.stackCount)

        self._dropStack(pyx.cast(pyx.int, i))

        #print('Stack Count after initial drop:', self.stackCount)

        xi:pyx.double = self.popStack()
        xi_n:pyx.double = self.popStack()

        yi:pyx.double   = i*dY
        yi_n:pyx.double = yi + dY

        #print(xi, ' -> ', xi_n)

        self._dropStack(pyx.cast(pyx.int, nBins - i - 2))

        #print('Stack Count after dropping remaining:', self.stackCount)

        m:pyx.double = (xi_n - xi) / (yi_n - yi)

        x_:pyx.double = xi + m*(y_ - yi)

        self.pushStack(x_)


        # dP: pyx.double  = 1. / nBins
        # v: pyx.double   = _rand()
        # som: pyx.double = 0

        # x_ = np.zeros(nBins)
        # y_ = np.zeros(nBins)
        # x: pyx.double[:] = x_
        # y: pyx.double[:] = y_

        # for i in range(nBins):
        #     x[i] = self.popStack()
        #     y[i] = dP*i

        # # We need to find the bucket that u is in.
        # if v == 0:
        #     pass
        #     # Return left point.
        # elif v == 1:
        #     pass
        #     # Retuen right point.
        # else:
        #     idx: pyx.double = c_floor(u / dP)

        #     # Find the containing bin.
        #     for i in range(nBins):
        #         if y[i] < v < y[i+1]:
        #             break







    def printState(self):
        _stack = []
        for i in reversed(range(self.stackCount)):
            _stack.append(self.stack[i])

        _stack = " ".join([f'{s:.0f}' for s in _stack ])

        print(f'{self.counter}: {self._codes[self.counter]:.0f}     {self._operands[self.counter]} -> [{_stack}]    {self.pointers[:2]}')





    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def sample(self) -> pyx.float:

        # for e, (c, o) in enumerate(list(zip(self.codes, self.operands))):
        #     print(f'{e}: {c}   {o}')

        N:pyx.int = self._codes.shape[0]
        #i:pyx.int = 0
        opCode: pyx.double
        operand: pyx.double

        while self.counter < N:
            #self.printState()
            opCode = self._codes[self.counter]
            operand = self._operands[self.counter]

            if   opCode == _PASS:
                pass
            elif opCode == _PUSH:
                self.pushStack(self._operands[self.counter])
            elif opCode == _STORE:
                self._store(self._operands[self.counter])
            elif opCode == _LOAD:
                self._load(self._operands[self.counter])
            elif opCode == _SUM_START:
                self._sumStart(self._operands[self.counter])
            elif opCode == _SUM_END:
                self._sumEnd(self._operands[self.counter])
            elif _ADD <= opCode <= _BINOPMAX:
                self._binop(opCode)

            elif opCode == _RANDINT:
                self._randInt()
            elif opCode == _RANDNORM:
                self._randNorm()
            elif opCode == _RAND_QUANTILES:
                self._randQuantiles(operand)
            elif opCode == _ARRAY_SUM:
                self._arraySum(operand)

            self.counter += 1

        return self.popStack()

    def compute(self, samples:pyx.int=10000, maxBins:pyx.int=32):
        rv = Digest(maxBins=maxBins)
        i: pyx.int
        for i in range(samples):
            x:pyx.float = self.sample()
            self.reset()
            rv._add(x, 1)

        return rv

    def run(self):
        return self.compute()

#==================================================================================================

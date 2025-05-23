import numpy as np
from .core import *
from .opcodes import *

class RandomVariable():
    def __init__(self, *args):
        self.children = list(args)

    def __add__(self, other):
        return ADD(self, other)

    def __sub__(self, other):
        return SUB(self, other)

    def __mul__(self, other):
        return MUL(self, other)

    def __pow__(self, other):
        return POW(self, other)

    def __truediv__(self, other):
        return DIV(self, other)

    def __mod__(self, other):
        return MOD(self, other)

    def __floordiv__(self, other):
        return FLOORDIV(self, other)

    def __divmod__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __rpow__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __rfloordiv__(self, other):
        pass

    def __rdivmod__(self, other):
        pass

    def printTree(self, level=0):
        print(' '*level*4+self.__class__.__name__)
        for c in self.children:
            if hasattr(c, 'printTree'):
                c.printTree(level+1)
            else:
                print(' '*(level + 1)*4+str(c))

    def _compile(self, codes, operands):
        pass

    def _compileOrPush(self, codes, operands, child):
        if hasattr(child, '_compile'):
            child._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(child)


    def _compileChildren(self, codes, operands):
        for c in self.children:
            self._compileOrPush(codes, operands, c)
            # if hasattr(c, '_compile'):
            #     c._compile(codes, operands)
            # else:
            #     codes.append(OP_PUSH)
            #     operands.append(c)

                # program.append(('PSH',c))

    def compile(self):
        codes    = []
        operands = []
        self._compile(codes, operands)
        print(codes)
        print(operands)
        codes = np.array(codes, dtype=np.double)
        operands = np.array(operands, dtype=np.double)
        return codes, operands

    def sample(self):
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        return vm.sample()

    def compute(self, samples=10000, maxBins=32):
        print('Compiling...')
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        print('Simulating...')
        return vm.compute(samples=samples, maxBins=maxBins)


class Constant(RandomVariable):
    def printTree(self,level=0):
        print(' '*level*4+str(self.children[0]))

    def _compile(self, codes, operands):
        codes.append(OP_PUSH)
        operands.append(self.children[0])
        # program.append(('PSH', self.children[0]))


class RandInt(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_RANDINT)
        operands.append(0)

class ADD(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_ADD)
        operands.append(0)

class SUB(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_SUB)
        operands.append(0)


class MUL(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MUL)
        operands.append(0)

class DIV(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_DIV)
        operands.append(0)

class FLOORDIV(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_FLOORDIV)
        operands.append(0)

class MOD(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MOD)
        operands.append(0)

class POW(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_POW)
        operands.append(0)

class NORMAL(RandomVariable):
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev

    def _compile(self, codes, operands):
        self._compileOrPush(codes, operands, self.mean)
        self._compileOrPush(codes, operands, self.stdev)
        # self._compileChildren(codes, operands)
        codes.append(OP_RANDNORM)
        operands.append(0)


class SUM(RandomVariable):
    """
    Creates a summation (Sigma symbol in mathematics) for loop. The first argument is the
    number of iterations and the second argument is the term that needs to be added
    together.
    """
    def __init__(self, nTerms=1, term=0):
        self.nTerms = nTerms
        self.term   = term
    def _compile(self, codes, operands):


        # Compile nTerms.
        if hasattr(self.nTerms, '_compile'):
            self.nTerms._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(self.nTerms)

        codes.append(OP_SUM_START)
        operands.append(0)

        if hasattr(self.term, '_compile'):
            self.term._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(self.term)

        codes.append(OP_SUM_END)
        operands.append(0)


#========================================================================================

class Quantiles(RandomVariable):
    """
    Samples from equally spaced quantile values.
    """
    def __init__(self, *args):
        self.children = list(reversed(args))

    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_RAND_QUANTILES)
        operands.append(len(self.children))

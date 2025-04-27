import numpy as np
from .core import *

class RandomVariable():
    def __init__(self, *args):
        self.children = list(args)

    def __add__(self, other):
        return ADD(self, other)

    def __mul__(self, other):
        return MUL(self, other)

    def __pow__(self, other):
        return POW(self, other)

    def __truediv__(self, other):
        pass

    def __floordiv__(self, other):
        pass

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

    def _compileChildren(self, codes, operands):
        for c in self.children:
            if hasattr(c, '_compile'):
                c._compile(codes, operands)
            else:
                codes.append(OP_PUSH)
                operands.append(c)

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

class MUL(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MUL)
        operands.append(0)


class POW(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_POW)
        operands.append(0)

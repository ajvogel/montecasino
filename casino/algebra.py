import numpy as np
from .core import VirtualMachine


for x in range(5):
    print(x)

__PASS__    = 0
__PUSH__    = 1
__ADD__     = 2
__MUL__     = 3
__POW__     = 4
__RANDINT__ = 5

class Node():
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
                codes.append(__PUSH__)
                operands.append(c)

                # program.append(('PSH',c))

    def compile(self):
        codes    = []
        operands = []
        self._compile(codes, operands)
        codes = np.array(codes)
        operands = np.array(operands)
        return codes, operands

    def compute(self):
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        return vm.run()

    def sample(self, samples=10000, maxBins=32):
        print('Compiling...')
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        print('Simulating...')
        return vm.sample(samples=samples, maxBins=maxBins)
        

class Num(Node):
    def printTree(self,level=0):
        print(' '*level*4+str(self.children[0]))

    def _compile(self, codes, operands):
        codes.append(__PUSH__)
        operands.append(self.children[0])
        # program.append(('PSH', self.children[0]))


class RandInt(Node):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(__RANDINT__)
        operands.append(0)

class ADD(Node):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(__ADD__)
        operands.append(0)

class MUL(Node):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(__MUL__)
        operands.append(0)        


class POW(Node):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(__POW__)
        operands.append(0)                


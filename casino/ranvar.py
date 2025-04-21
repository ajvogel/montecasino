import numpy as np
from .digest import TDigest
from .engine import Engine


for x in range(5):
    print(x)



class RandomVariable():
    def __init__(self, *args):
        self.children = list(args)

    def __add__(self, other):
        return ADD(self, other)

    def __mul__(self, other):
        return MUL(self, other)

    def __sub__(self, other):
        return SUB(self, other)

    def __div__(self, other):
        return DIV(self, other)

    def __pow__(self, other):
        return POW(self, other)

    def printTree(self, level=0):
        print(' '*level*4+self.__class__.__name__)
        for c in self.children:
            if hasattr(c, 'printTree'):
                c.printTree(level+1)
            else:
                print(' '*(level + 1)*4+str(c))

    def _compileWithChildren(self, codes, operands):
        self._compileChildren(codes, operands)
        self._compile(codes, operands)

    def _compile(self, codes, operands):
        raise NotImplementedError

    def _compileChildren(self, codes, operands):
        for c in self.children:
            if hasattr(c, '_compileWithChildren'):
                c._compileWithChildren(codes, operands)
            elif hasattr(c, '_compile'):
                c._compile(codes, operands)
            else:
                codes.append(Engine.__PUSH__)
                operands.append(c)

    def compile(self):
        codes    = []
        operands = []
        self._compile(codes, operands)
        codes = np.array(codes)
        operands = np.array(operands, dtype=np.double)
        return codes, operands

    def compute(self):
        codes, operands = self.compile()
        vm = Engine(codes, operands)
        return vm.compute()

class RandomVariableDigest(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__DIGEST__)
        operands.append(0)

class Num(RandomVariable):
    def printTree(self,level=0):
        print(' '*level*4+str(self.children[0]))

    def _compile(self, codes, operands):
        codes.append(Engine.__PUSH__)
        operands.append(self.children[0])
        # program.append(('PSH', self.children[0]))


class RandInt(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__RANDINT__)
        operands.append(0)

class ADD(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__ADD__)
        operands.append(0)


class SUB(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__SUB__)
        operands.append(0)

class DIV(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__DIV__)
        operands.append(0)

class MUL(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__MUL__)
        operands.append(0)


class POW(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(Engine.__POW__)
        operands.append(0)

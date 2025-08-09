from typing import SupportsAbs
import numpy as np
from .core import *
from .opcodes import *

class RandomVariable():
    def __init__(self, *args):
        self.children = list(args)

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __floordiv__(self, other):
        return FloorDiv(self, other)

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



#-----------------------------------------------------------------------------------------

class DigestVariable(RandomVariable):
    """
    Wraps the Digest class in a RandomVariable class so that we can use it in our
    algebra operations.
    """
    def __init__(self, digest: Digest):
        self._digest = digest

    def _compile(self, codes, operands):
        # We convert and compile the digest node into a generic histogram for random
        # sampling.
        
        x = self._digest.getBins()
        w = self._digest.getWeights()
        n = self._digest.getActiveBinCount()

      

        # Trim possible zeros at the end of the arrays.
        x = x[:n]
        w = w[:n]

        print([f'{xx:.1f}' for xx in x])
        print([f'{ww:.1f}' for ww in w])          

        b = np.zeros(n - 1)


        
        c = np.zeros(n)

        for i in range(n - 1):
            b[i] = (w[i] + w[i+1]) / 2

        for i in range(n):
            for j in range(0, i - 1):
                c[i] = c[i] + b[j]

        c = c / b.sum()

        print("Bins:")

        print([f'{bb:.1f}' for bb in b])
        print("C:")
        print([f'{cc:.5f}' for cc in c])
        c2 = b.cumsum() / b.sum()        
        print([f'{cc:.5f}' for cc in c2])        

        c[1:] = c2
        c[0]  = 0
        print([f'{cc:.5f}' for cc in c])
        for i in range(n - 1, -1, -1):
            self._compileOrPush(codes, operands, c[i])
            self._compileOrPush(codes, operands, x[i])

        codes.append(OP_RAND_HIST)
        operands.append(n)
        
            
        
        
    
#-----------------------------------------------------------------------------------------

class Constant(RandomVariable):
    def printTree(self,level=0):
        print(' '*level*4+str(self.children[0]))

    def _compile(self, codes, operands):
        codes.append(OP_PUSH)
        operands.append(self.children[0])


#---------------------------------------------------------------------------------------------------


class RandInt(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_RANDINT)
        operands.append(0)

#---------------------------------------------------------------------------------------------------

class Add(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_ADD)
        operands.append(0)

#---------------------------------------------------------------------------------------------------        

class Sub(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_SUB)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                

class Mul(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MUL)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                        

class Div(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_DIV)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                                

class FloorDiv(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_FLOORDIV)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                                        

class Mod(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MOD)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                                                

class Pow(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_POW)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                                                

class Normal(RandomVariable):
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev

    def _compile(self, codes, operands):
        self._compileOrPush(codes, operands, self.mean)
        self._compileOrPush(codes, operands, self.stdev)
        # self._compileChildren(codes, operands)
        codes.append(OP_RANDNORM)
        operands.append(0)

#---------------------------------------------------------------------------------------------------                                                

class Summation(RandomVariable):
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

#---------------------------------------------------------------------------------------------------                                                

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


#---------------------------------------------------------------------------------------------------                                                
        
class ArraySum(RandomVariable):
    """
    Stochastically sums together a contiguous list of random variables.
    """
    def __init__(self, array, start, end):
        self.children = list(reversed(array))
        self.start = start
        self.end = end

    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        self._compileOrPush(codes, operands, self.end)
        self._compileOrPush(codes, operands, self.start)

        codes.append(OP_ARRAY_SUM)
        operands.append(len(self.children))

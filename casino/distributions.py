import numpy as np
from .core import RandomVariable


#---------------------------------------------------------------------------------------------------

class Uniform(RandomVariable):
    def __init__(self, left, right) -> None:
        self.left  = left
        self.right = right

        self.p = 1. / ((self.right - self.left) + 1)


    def pmf(self, k):
        if self.left <= k <= self.right:
            return self.p
        else:
            return 0

    def lowerBound(self):
        return self.left

    def upperBound(self):
        return self.right


#---------------------------------------------------------------------------------------------------


class Triangular(RandomVariable):
    def __init__(self, left, mode, right, endpoints=False):

        if endpoints:
            self.left  = left - 1
            self.right = right + 1
        else:
            self.left = left
            self.right = right

        self.mode = mode

    def pmf(self, k):

        if k <= self.left:
            return 0
        elif self.left < k <= self.mode:
            return 2*(k - self.left) / ((self.right - self.left)*(self.mode - self.left))
        elif k == self.mode:
            return 2/(self.right - self.left)
        elif self.mode < k <= self.right:
            return 2*(self.right - k) / ((self.right - self.left)*(self.right - self.mode))
        else:
            return 0

    def lowerBound(self):
        return self.left

    def upperBound(self):
        return self.right

#---------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    t1 = ThreePointEstimation(10,50,20)
    t1.hist()

    # dice1 = Dice()
    # bla = dice1 

    bla = t1

    for i in range(10):
        bla = bla + ThreePointEstimation(10, 50, 20)

    bla.compress()

    bla.hist()


    # for i in range(10):
    #     print(f'Iteration {i}')

    #     bla = bla + Dice()
    #     print(bla.k)
    #     print(bla.w)        
    #     print(bla.k.min())

    # bla.hist()
    # print(bla.w)
    # print(bla.k)

    # bla.compress()

    # bla.hist()
    # print(bla.w)
    # print(bla.k)    
    # print(bla.k.min())


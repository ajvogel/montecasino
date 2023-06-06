import numpy as np
import numba

LOWER = 0.001
UPPER = 1 - LOWER

__ADD__ = 0
__MUL__ = 1

#---------------------------------------------------------------------------------------------------

class RandomVariable():
    def cdf(self, k):
        prob = 0
        for i in range(self.min(), k + 1):
            prob += self.pmf(i)

        return prob

    def pmf(self, k):
        return 0

    def min(self):
        return 0

    def toArray(self):
        outW = []
        outK = []

        som = 0
        k = self.min()
        while som < UPPER:
            pk = self.pmf(k)
            som += pk
            k   += 1
            outW.append(pk)
            outK.append(k)

        return np.array(outW), np.array(outK)

    def __conv__(self, other, func):
        out = DiscreteEmperical()
        # We extract the probability array once in the beginning of the loop
        # so that we don't have to regenerate it everytime. We are essentially
        # iterating over the same array more than once.
        pSelf, kSelf   = self.toArray()
        pOther, kOther = other.toArray()

        sMin = self.min()
        oMin = other.min()

        nS = pSelf.shape[0]
        nO = pOther.shape[0]

        for iS in range(nS):

            for iO in range(nO):
                sk = sMin + iS
                ok = oMin + iO
                fp = pSelf[iS] * pOther[iO]

                if func == __ADD__:
                    fk = ok + sk
                elif func == __MUL__:
                    fk = ok*sk

                out.add(fk, fp)

        out.compress()
        return out


    def __add__(self, other):
        return self.__conv__(other, __ADD__)

    def __mul__(self, other):
        return self.__conv__(other, __MUL__)
      


#---------------------------------------------------------------------------------------------------

class Dice(RandomVariable):

    def pmf(self, k):
        if 1 <= k <= 6:
            return 1./6
        else:
            return 0

    def min(self):
        return 1

    def max(self):
        return 7


#---------------------------------------------------------------------------------------------------


class Triangular(RandomVariable):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def pmf(self, k):

        if k <= self.a:
            return 0
        elif self.a < k <= self.c:
            return 2*(k - self.a) / ((self.b - self.a)*(self.c - self.a))
        elif k == self.c:
            return 2/(self.b - self.a)
        elif self.c < k <= self.b:
            return 2*(self.b - k) / ((self.b - self.a)*(self.b - self.c))
        else:
            return 0

    def min(self):
        return self.a

    def hist(self):

        maxW = self.pmf(self.c)

        cumProb = 0
        k = self.min()
        while cumProb < UPPER:
            p = self.pmf(k)
            print(f'{k:5} | {"█"*int(p/maxW*50)}')
            cumProb += self.pmf(k)
            k += 1            

#---------------------------------------------------------------------------------------------------

class ThreePointEstimation(Triangular):
    def __init__(self, worst=1, best=10, most=5):
        super().__init__(worst-1, best+1, most)


#---------------------------------------------------------------------------------------------------

@numba.njit
def _bisection(kk, k):
    nK = kk.shape[0]
    u  = nK
    l  = 0

    if nK == 0:
        return None

    while True:
        m = (u + l) // 2

        if kk[m] == k:
            return m
        elif kk[m] < k:
            l = m
        elif kk[m] > k:
            u = m

        if (u - l) <= 1:
            return None

#---------------------------------------------------------------------------------------------------

class DiscreteEmperical(RandomVariable):
    def __init__(self) -> None:
        self.k = np.array([], dtype=np.int32)
        self.w = np.array([], dtype=np.float32)

    def toArray(self):
        return self.w, self.k        

    def min(self):

        if len(self.k) == 0:
            return 0
        else:
            return self.k.min()

    def add(self, k, weight=1):
        """
        Adds another sampel to this distribution.
        """
        if weight == 0:
            return


        if ik := _bisection(self.k, k):   
            self.w[ik] += weight
        else:
            self.k = np.append(self.k, k)
            self.w = np.append(self.w, weight)
            idx = np.argsort(self.k)
            self.k = self.k[idx]
            self.w = self.w[idx]            

    def pmf(self, k):
        kk   = self.k
        ww   = self.w
        wSum = ww.sum()
        nK   = len(kk)

        # We now that the k and w arrays are sorted, so we can use a bisection search
        # to find the correct values faster. Iterating over every thing is probably
        # inefficent.

        if ik := _bisection(self.k, k):
            return ww[ik] / wSum
        else:
            return 0


    def compress(self):
        """
        Comresses the distribution by dropping the tail entries.
        """
        cumProb = 0

        minI = 9e9
        maxI = -9e9

        for i in range(len(self.k)):

            k = self.k[i]
            cumProb += self.pmf(k)

            if LOWER <= cumProb <= UPPER:
                minI = min(i, minI)
                maxI = max(i, maxI)

        self.k = self.k[minI:maxI+1]
        self.w = self.w[minI:maxI+1]









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


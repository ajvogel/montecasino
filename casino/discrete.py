import numpy as np

LOWER = 0.001
UPPER = 1 - LOWER

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

    def hist(self):

        cumProb = 0
        k = self.min()
        while cumProb < UPPER:
            p = self.pmf(k)
            print(f'{k:5} | {"#"*int(p*1000)}')
            cumProb += self.pmf(k)
            k += 1


    def __add__(self, other):

        out = DiscreteEmperical()

        sCum = 0
        sk = self.min()
        while sCum < UPPER:

            oCum = 0
            ok = other.min()
            while oCum < UPPER:

                fk = ok + sk
                fp = self.pmf(sk) * other.pmf(ok)

                out.add(fk, fp)

                oCum += self.pmf(ok)
                ok   += 1

            sCum += self.pmf(sk)
            sk   += 1

        return out

    def __mul__(self, other):

        out = DiscreteEmperical()

        sCum = 0
        sk = self.min()
        while sCum < UPPER:

            oCum = 0
            ok = other.min()
            while oCum < UPPER:

                fk = ok * sk
                fp = self.pmf(sk) * other.pmf(ok)

                out.add(fk, fp)

                oCum += self.pmf(ok)
                ok   += 1

            sCum += self.pmf(sk)
            sk   += 1

        return out        


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

class DiscreteEmperical(RandomVariable):
    def __init__(self) -> None:
        self.k = np.array([], dtype=np.int32)
        self.w = np.array([], dtype=np.float32)

    def min(self):

        if len(self.k) == 0:
            return 0
        else:
            print(f'Returning min = {self.k.min()}')
            return self.k.min()

    def add(self, k, weight=1):
        """
        Adds another sampel to this distribution.
        """
        if weight == 0:
            return


        for i in range(len(self.k)):
            # TODO: Replace this with bisection searching, which should be more efficient.
            if self.k[i] == k:
                self.w[i] += weight
                break
        else:
            # The current k value is not already in the data arrays.
            self.k = np.append(self.k, k)
            self.w = np.append(self.w, weight)
            idx = np.argsort(self.k)
            self.k = self.k[idx]
            self.w = self.w[idx]

    def hist(self):

        maxW = self.w.max()

        cumProb = 0
        k = self.min()
        while cumProb < UPPER:
            p = self.pmf(k)
            print(f'{k:5} | {"█"*int(p / maxW *50)}')
            cumProb += self.pmf(k)
            k += 1            

    def pmf(self, k):
        for i in range(len(self.k)):
            if self.k[i] == k:
                return self.w[i] / self.w.sum()
        else:
            return 0.


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

    dice1 = Dice()
    bla = dice1 

    for i in range(10):
        print(f'Iteration {i}')

        bla = bla + Dice()
        print(bla.k)
        print(bla.w)        
        print(bla.k.min())

    bla.hist()
    print(bla.w)
    print(bla.k)

    bla.compress()

    bla.hist()
    print(bla.w)
    print(bla.k)    
    print(bla.k.min())

    # dice1 = Dice()
    # dice2 = Dice()
    # dice3 = Dice()
    # dice4 = Dice()
    # dice5 = Dice()

    # dice1.hist()

    # bla = dice1 + dice2 + dice3 + dice4 + dice5

    # bla2 = bla + dice5

    # print(bla.k)
    # print(bla.w)
    # print(bla.w.sum())

    # print(bla2.k)
    # print(bla2.w)
    # print(bla2.w.sum())

    # bla.hist()
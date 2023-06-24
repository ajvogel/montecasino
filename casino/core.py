import numpy as np
import numba

LOWER = 0.001
UPPER = 1 - LOWER

__ADD__ = 0
__MUL__ = 1
__MAX__ = 2
__MIN__ = 3
__SUB__ = 4
__POW__ = 5

#===================================================================================================

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
        
#===================================================================================================

class RandomVariable():
    def cdf(self, k):
        prob = 0
        for i in range(self.lowerBound(), k + 1):
            prob += self.pmf(i)

        return prob

    def pmf(self, k):
        return 0

    def lowerBound(self):
        return 0

    def toArray(self):
        outW = []
        outK = []

        som = 0
        k = self.lowerBound()
        while som < UPPER:
            pk = self.pmf(k)
            som += pk
            k   += 1
            outW.append(pk)
            outK.append(k)

        return np.array(outW), np.array(outK)

    def __conv__(self, other, func):
        out = Empirical()
        # We extract the probability array once in the beginning of the loop
        # so that we don't have to regenerate it everytime. We are essentially
        # iterating over the same array more than once.
        pSelf, kSelf   = self.toArray()
        pOther, kOther = other.toArray()

        sMin = self.lowerBound()
        oMin = other.lowerBound()

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
                elif func == __MAX__:
                    fk = ok if ok > sk else sk
                elif func == __MIN__:
                    fk = ok  if ok < sk else sk
                elif func == __SUB__:
                    fk = sk - ok
                elif func == __POW__:
                    fk = sk**ok

                out.add(fk, fp)

        out.compress()
        return out


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


#===================================================================================================


class Empirical(RandomVariable):
    def __init__(self) -> None:
        self.k = np.array([], dtype=np.int32)
        self.w = np.array([], dtype=np.double)

    def toArray(self):
        return self.w, self.k        

    def lowerBound(self):

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
            # TODO: Replace this algorithm with an algorithm that makes better use of the fact
            #       that we already have a sorted list and we are only inserting a single entry.
            self.k = np.append(self.k, k)
            self.w = np.append(self.w, weight)
            idx = np.argsort(self.k)
            self.k = self.k[idx]
            self.w = self.w[idx]            

    def pmf(self, k):
        kk   = self.k
        ww   = self.w
        wSum = ww.sum()

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

        if self.w.shape[0] <= 100:
            return


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

        # Some opperations such as * cause exponentially increasing k vectors,
        # which causes things to scale very badly. We need a way to limit the
        # increase here so that we can atleast implement some form of upper
        # limit on the CPU time.

        if self.w.shape[0] > 500:
            idx = np.argsort(self.w)[::-1]
            self.w = self.w[idx][:500]
            self.k = self.k[idx][:500]

            # Sort back in ascending k order for things to work.
            idx = np.argsort(self.k)
            self.w = self.w[idx]
            self.k = self.k[idx]        

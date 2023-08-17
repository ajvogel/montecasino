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

DEFAULTS = {
    'maxBins':64
}

#===================================================================================================

class RandomVariable():
    def __init__(self, maxBins=None):
        if maxBins is None:
            maxBins = DEFAULTS['maxBins']

        self.lower   = np.zeros(maxBins)
        self.upper   = np.zeros(maxBins)
        self.freq    = np.zeros(maxBins)
        self.known   = np.zeros(maxBins)
        self.unknown = np.zeros(maxBins)

        self.nActive = 0
        self.maxBins = maxBins

    def lowerBound(self):
        return self.lower[0]

    def upperBound(self):
        return self.upper[-1]

    def cdf(self, k):
        out = 0
        for i in range(self.nActive):
            if self.upper[i] <= k:
                # K is higher than the current bin.
                out += self.freq[i] / self.freq.sum()

            elif self.lower[i] <= k < self.upper[i]:
                #  k is within the bin.
                w = self.upper[i] - self.lower[i]
                p = self.freq[i] / (self.freq.sum() * w)
                out += p*(k - self.lower[i])

            else:
                # k is under the bin.
                break

        return out


    def pmf(self, k):
        for i in range(len(self.upper)):
            if self.lower[i] <= k < self.upper[i]:
                # Each point in the bin has the same probability.
                w = self.upper[i] - self.lower[i]
                p = self.freq[i] / (self.freq.sum() * w)
                return p
        else:
            return -1

    def _sortBins(self):
        idx = np.argsort(self.lower)

        self.lower   = self.lower[idx]
        self.upper   = self.upper[idx]
        self.known   = self.known[idx]
        self.unknown = self.unknown[idx]

    def _addPhaseOne(self, k, weight):
        """
        Adds a point when we haven't already filled all the different histograms.
        """
        for i in range(self.nActive):
            if self.lower[i] <= k < self.upper[i]:
                self.freq[i]  += weight
                self.known[i] += weight
                break
        else:
            i = self.nActive

            self.lower[i] = k
            self.upper[i] = k + 1
            self.freq[i]  = weight
            self.known[i] = weight

            self.nActive += 1

            if self.nActive == self.maxBins:
                self._sortBins()

    def _merge(self, iMin):
        # Stretches the iMin bin to encompass the iMin+1 bin as well. This clears
        # up the iMin+1 bin to use for splitting.
        self.upper[iMin] = self.upper[iMin + 1]

        self.freq[iMin]    = self.freq[iMin]    + self.freq[iMin + 1]
        self.known[iMin]   = self.known[iMin]   + self.known[iMin + 1]
        self.unknown[iMin] = self.unknown[iMin] + self.unknown[iMin + 1]

    def _split(self, iMax, m1):
        # Splits iMax into two bins and stores the one in m1.
        l = self.lower[iMax]
        u = self.upper[iMax]
        w = u - l

        fa = self.freq[iMax]
        fk = self.known[iMax]
        fu = self.unknown[iMax]

        m2 = iMax

        self.lower[m1] = l
        self.upper[m1] = l + w/2
        self.known[m1] = 0
        self.unknown[m1] = fk + fu
        self.freq[m1] = fa / 2


        self.lower[m2] = l + w/2
        self.upper[m2] = u
        self.known[m2] = 0
        self.unknown[m2] = fk + fu        
        self.freq[m2] = fa / 2        


    def add(self, k, weight=1):

        if self.nActive < self.maxBins:
            self._addPhaseOne(k, weight=weight)
        else:

            costLower = (self.upper - self.lower) * (self.known - self.unknown)
            costUpper = (self.upper - self.lower) * (self.known + self.unknown)

            adjCostUpper = costUpper[:-1] + costUpper[1:]

            iMaxLower = np.argmax(costLower)
            iMinUpper = np.argmin(adjCostUpper)

            if costLower[iMaxLower] > 2*costUpper[iMinUpper]:
                self._merge(iMinUpper)
                self._split(iMaxLower, iMinUpper + 1)
                self._sortBins()




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
        # out = SECHAP()
        # out = ECHAP()
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



#======================================================================================
@numba.njit
def _frequency(bl, bu, data):
    n = len(bl)
    f = np.zeros(bl.shape)

    for di in data:
        for i in range(n):
            if bl[i] <= di < bu[i]:
                f[i] += 1
                break
        else:
            f[-1] += 1

    return f


class ECHAP(RandomVariable):
    """
    Vermorel and Bronnimann, ‘Greedy Online Histograms Applied to Deterministic Sampling’.
    """
    def __init__(self, nBins=16):
        self.nBins = nBins

        self.bl = np.zeros(nBins)
        self.bu = np.zeros(nBins)
        self.f  = np.zeros(nBins)

    def lowerBound(self):
        return self.bl[0]

    def upperBound(self):
        return self.bu[-1]

    def pmf(self, k):
        i = np.where(self.bl >= k)[0][0]
        w = self.bu[i] - self.bl[i]
        p = self.f[i] / (self.f.sum() * w)
        return p


    def fit(self, data):
        data = np.sort(data)
        minD = data.min()
        maxD = data.max()
        bins = np.linspace(minD, maxD, self.nBins + 1)

        bl = bins[:-1].copy()
        bu = bins[1:].copy()

        while True:
            f  = _frequency(bl, bu, data)
            c  = (bu - bl) * f
            cc = c[:-1] + c[1:]

            iMax = np.argmax(c)
            iMin = np.argmin(cc)

            if c[iMax] <= 2*cc[iMin]:
                break

            # Merge (iMin) and (iMin + 1)
            bu[iMin] = bu[iMin + 1]

            # Split iMax
            m1 = iMin + 1 # Available from the merge.

            wMax = bu[iMax] - bl[iMax]
            bl[m1] = bl[iMax]
            bu[m1] = bl[iMax] + wMax/2

            bl[iMax] = bu[m1] # bu[iMax] is unchanged.

            # Sorting the bins again. Possible to avoid the shuffling if we are smarter
            # with how we do the splitting and merging above.

            idx = np.argsort(bl)
            bl  = bl[idx]
            bu  = bu[idx]
            f   = f[idx]

        self.f  = f
        self.bl = bl
        self.bu = bu



#======================================================================================

@numba.njit
def _addPointToBin(k, w, bl, bu, fa, fk):
    if k < bl[0]:
        # We need to stretch the lower in.
        bl[0]  = k
        fa[0]  += w
        fk[0] += w
    elif k > bu[-1]:
        # We need to stretch the upper bin.
        bu[-1]  = k
        fa[-1]  += w
        fk[-1] += w
    else:
        # i = np.where(bl <= k)[0][-1]
        i = _findBin(bl, bu, k)
        # fa[i] = fa[i] + w
        fa[i] += w
        fk[i] += w    

    return bl, bu, fa, fk

@numba.njit
def _argmax(vec):
    maxVal = -99999999
    maxI   = -1
    for i in range(len(vec)):
        if vec[i] > maxVal:
            maxVal = vec[i]
            maxI   = i

    return maxI

@numba.njit
def _argmin(vec):
    maxVal = 99999999
    minI   = -1
    for i in range(len(vec)):
        if vec[i] < maxVal:
            maxVal = vec[i]
            minI   = i

    return minI    

@numba.njit
def _findBin(bl, bu, k: int) -> numba.int64:
    for i in range(len(bl)):
        if bl[i] <= k < bu[i]:
            return i
    return -1


@numba.njit
def _add(bl, bu, fa, fk, fu):
    c_n = (bu - bl)*(fk - fu)
    c_p = (bu - bl)*(fk + fu)

    cc_p = c_p[:-1] + c_p[1:]

    # iMax = np.argmax(c_n)
    # iMin = np.argmin(cc_p)

    iMax = _argmax(c_n)
    iMin = _argmin(cc_p)

    if c_n[iMax] > 2*cc_p[iMin]:

        # Merge (iMin) and (iMin + 1)
        bu[iMin] = bu[iMin + 1]
        fk[iMin] = fk[iMin] + fk[iMin + 1]
        fu[iMin] = fu[iMin] + fu[iMin + 1]
        fa[iMin] = fa[iMin] + fa[iMin + 1]

        # Split iMax
        m1 = iMin + 1 # Available from the merge.

        # Saving these values because the change in lower ops.
        wMax  = bu[iMax] - bl[iMax]
        fkMax = fk[iMax]
        fuMax = fu[iMax]
        faMax = fa[iMax]

        bl[m1] = bl[iMax]
        bu[m1] = bl[iMax] + wMax/2

        bl[iMax] = bu[m1] # bu[iMax] is unchanged.           

        fk[iMax] = 0
        fk[m1]   = 0

        fu[iMax] = fkMax + fuMax
        fu[m1]   = fkMax + fuMax
        fa[iMax] = faMax / 2
        fa[m1]   = faMax / 2

        # Sorting the bins again. Possible to avoid the shuffling if we are smarter
        # with how we do the splitting and merging above.

        idx = np.argsort(bl)
        bl  = bl[idx]
        bu  = bu[idx]
        fa  = fa[idx]            
        fk  = fk[idx]            
        fu  = fu[idx]                
    return bl, bu, fa, fk, fu

class SECHAP(RandomVariable):
    def __init__(self, nBins=16):
        self.nBins = nBins

        self.bl = np.zeros(nBins)
        self.bu = np.zeros(nBins)
        self.fa = np.zeros(nBins)        
        self.fk = np.zeros(nBins)        
        self.fu = np.zeros(nBins)        

        self.activeBins = 0

    def compress(self):
        pass

    def lowerBound(self):
        return self.bl[0]

    def upperBound(self):
        return self.bu[-1]

    def pmf(self, k):
        # print(f'pmf({k})')
        # print(np.where(self.bl <= k))
        i = _findBin(self.bl, self.bu, k)
        # i = np.where(self.bl <= k)[0][-1]
        w = self.bu[i] - self.bl[i]
        p = self.fa[i] / (self.fa.sum() * w)
        return p


    def _addPhaseOne(self, k, w):
        """

        """
        for i in range(self.activeBins):
            if self.bl[i] == k:
                self.fa[i]  += w
                self.fk[i] += w
                break
        else:
            
            i = self.activeBins
            self.bl[i] = k
            self.fa[i] = w
            self.fk[i] = w
            self.activeBins += 1

            if self.activeBins == self.nBins:
                # We've filled all the bins now. Clean up before going to
                # phase 2.
                idx = np.argsort(self.bl)
                self.bl = self.bl[idx]
                self.fa = self.fa[idx]
                self.fk = self.fk[idx]

                # No need to sort the others because by this point they shouldn't
                # have any values in.

                self.bu[:-1] = self.bl[1:]
                self.bu[-1]  = self.bl[-1] + 1

                # for ll, uu, ff in zip(self.bl, self.bu, self.fa):
                #     print(f'{ll}..{uu} -> {ff}') 

    def _addPointToBin(self, k, w):
        # bl, bu, fa, fk = (self.bl, self.bu, self.fa, self.fk)
        # self.bl, self.bu, self.fa, self.fk = _addPointToBin(k, w, bl, bu, fa, fk)
        if k < self.bl[0]:
            # We need to stretch the lower in.
            self.bl[0]  = k
            self.fa[0]  += w
            self.fk[0] += w
        elif k > self.bu[-1]:
            # We need to stretch the upper bin.
            self.bu[-1]  = k
            self.fa[-1]  += w
            self.fk[-1] += w
        else:
            # i = np.where(self.bl <= k)[0][-1]
            i = _findBin(self.bl, self.bu, k)
            self.fa[i]  += w
            self.fk[i] += w

    def fit(self, data):
        for d in data:
            self.add(d)

    def add(self, k, weight=1):
        k = round(k) 
        if self.activeBins < self.nBins:
            self._addPhaseOne(k, weight)
            return

        self._addPointToBin(k,weight)

        bl, bu, fa, fk, fu = (self.bl, self.bu, self.fa, self.fk, self.fu)

        # bl, bu, fa, fk = _addPointToBin(k, weight, bl, bu, fa, fk)

        # self.bl, self.bu, self.fa, self.fk, self.fu = _add(bl, bu, fa, fk, fu)


        c_n = (bu - bl)*(fk - fu)
        c_p = (bu - bl)*(fk + fu)

        cc_p = c_p[:-1] + c_p[1:]

        # iMax = np.argmax(c_n)
        # iMin = np.argmin(cc_p)

        iMax = _argmax(c_n)
        iMin = _argmin(cc_p)

        if c_n[iMax] > 2*cc_p[iMin]:

            # Merge (iMin) and (iMin + 1)
            bu[iMin] = bu[iMin + 1]
            fk[iMin] = fk[iMin] + fk[iMin + 1]
            fu[iMin] = fu[iMin] + fu[iMin + 1]
            fa[iMin] = fa[iMin] + fa[iMin + 1]

            # Split iMax
            m1 = iMin + 1 # Available from the merge.

            # Saving these values because the change in lower ops.
            wMax  = bu[iMax] - bl[iMax]
            fkMax = fk[iMax]
            fuMax = fu[iMax]
            faMax = fa[iMax]

            bl[m1] = bl[iMax]
            bu[m1] = bl[iMax] + wMax/2

            bl[iMax] = bu[m1] # bu[iMax] is unchanged.           

            fk[iMax] = 0
            fk[m1]   = 0

            fu[iMax] = fkMax + fuMax
            fu[m1]   = fkMax + fuMax
            fa[iMax] = faMax / 2
            fa[m1]   = faMax / 2

            # Sorting the bins again. Possible to avoid the shuffling if we are smarter
            # with how we do the splitting and merging above.

            idx = np.argsort(bl)
            bl  = bl[idx]
            bu  = bu[idx]
            fa  = fa[idx]            
            fk  = fk[idx]            
            fu  = fu[idx]            

        self.bl = bl
        self.bu = bu
        self.fa = fa
        self.fk = fk
        self.fu = fu

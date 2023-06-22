import numpy as np

A = 0
K = 1
U = 2
AK = [0, 1]

LOWER = 0
UPPER = 1


class Histogram():
    def __init__(self, nBins=128) -> None:
        self.nBins = nBins

        self.bins = np.zeros(nBins + 1) # We need one extra for the bin boundaries.
        self.activeBins  = 0 # The first non-zero index, useful when we have fewer points than nBins.

        self.F = np.zeros((nBins,3))

        self.total = 0

    def _printBins(self):
        for i in range(self.nBins):
            print(f'[{self.bins[i]:.0f}, {self.bins[i+1]:.0f}) -> ', end='')
        print()

    def _updatePhaseOne(self, v):
        """
        Performs an update when we haven't filled all the initial bins yet.
        """
        for i in range(self.activeBins):
            if self.bins[i] == v:
                self.F[i, AK] += 1
                break
        else:
            self.bins[self.activeBins]  = v
            self.F[self.activeBins, AK] += 1
            self.activeBins += 1

            if self.activeBins == self.nBins:
                idx = np.argsort(self.bins[:-1])
                self.bins[:-1] = self.bins[idx]
                self.F = self.F[idx, :]
                self.bins[-1] = self.bins[-2] + 1

    def _addPointToBin(self, v):
        if v > self.bins[-1]:   # Stretch upper bin to include point.
            self.bins[-1] = v + 1
            self.F[-1, AK] += 1
        elif v < self.bins[0]:  # Stretch lower bin to include point.
            self.bins[0] = v
            self.F[0, AK] += 1
        else:
            # Index of the bin that contains this value.

            idx = np.where(self.bins <= v)[0][-1]
            # print(self.F)
            self.F[idx, AK] += 1

    def _splitMaximumCostBin(self, B, F, bMin, bMax):
        b1 = bMax
        b2 = bMin + 1

        # We don't know exactly which bins are contain the data points so we sit 
        # fk (known) to zero.
        fu2 = F[bMax, K] + F[bMax, U]

        F[b1, K] = 0
        F[b2, K] = 0
        
        F[b1, U] = fu2
        F[b2, U] = fu2

        fa2 = F[bMax, A] / 2

        F[b1, A] = fa2
        F[b2, A] = fa2

        # fa[b1] = fa[bMax] / 2
        # fa[b2] = fa[bMax] / 2

        w = (B[bMax, UPPER] - B[bMax, LOWER]) / 2

        bu1 = B[b1, UPPER]
        bl1 = B[b1, LOWER]
        bm1 = bl1 + w

        # print(f'bl1 = {bl1}; bu1 = {bu1}; bm1 = {bm1}; w = {w}')

        B[b1, LOWER] = bl1
        B[b1, UPPER] = bm1
        B[b2, LOWER] = bm1
        B[b2, UPPER] = bu1

        return B, F
    

    def update(self, v):
        """

        bins = [0, 5, 10, 15, 20]
        fk   = [  2  4   6   8]
        
        """
        print("")
        print(f' => Received {v}...')
        if self.activeBins < self.nBins:
            self._updatePhaseOne(v)
            print(self.F[:,A].sum())
            print(self.bins)            
            # print(self.bins)
            return

        self._addPointToBin(v)

        F = self.F
        B = np.zeros((self.nBins, 2))
        B[:, LOWER] = self.bins[:-1]
        B[:, UPPER] = self.bins[1:]

        widths = B[:, UPPER] - B[:, LOWER]

        C = np.zeros((self.nBins, 2))

        C[:, LOWER] = (F[:,K] - F[:,U])*widths
        C[:, UPPER] = (F[:,K] + F[:,U])*widths

        bMax = np.argmax(C[:, LOWER])
        bMin = np.argmin(C[:-1, UPPER] + C[1:, UPPER])

        if C[bMax, LOWER] > 2*C[bMin, UPPER]:
            # print(' -> Splitting and Merging.')
            # print(f'bMax = {bMax}')
            # print(f'bMin = {bMin}')
            # print(f'bMin + 1 = {bMin + 1}')
            # print("Before changes:")
            # print(B)
            # If the maximum lower estimate is higher then twice the smallest consequtive
            # bin cost, then we should reduce the overall cost by splitting and merging
            # bins.

            # Merge bins.
            F[bMin, :] = F[bMin, :] + F[bMin + 1, :]
            B[bMin, UPPER] = B[bMin + 1, UPPER]

            # print('After merge:')
            # print(B)

            # After merging the entries in the bMin + 1 entry in our arrays become
            # available to use for splitting the large bin.

            # To split the bins we are going to resize bin bMax by half, and create
            # a new splitted bin in (bMin + 1) that has now become available.

            B, F = self._splitMaximumCostBin(B, F, bMin, bMax)

            # print('After split:')
            # print(B)
            # Rearranging the bins so that they are in the right order.
            idx = np.argsort(B[:, LOWER])
            B = B[idx, :]
            F = F[idx, :]

            print('After Sort:')
            print("B:")
            print(B)
            print("F:")
            print(F)

            # Backing back the bins in to the bins list.
            self.bins[:-1] = B[:, LOWER]
            self.bins[-1]  = B[-1, UPPER]

            # self.bins[:-1] = bl
            # self.bins[-1]  = bu[-1]

        print(self.F[:,A].sum())
        # print(self.bins)
        self._printBins()



if __name__ == "__main__":
    hist = Histogram(32)

    V = np.random.normal(scale=100, size=100000)

    for v in V:
        hist.update(v)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()

    w = hist.bins[1:] - hist.bins[:-1]

    print(w)
    ax.bar(hist.bins[:-1], hist.F[:,0], width=w, align='edge')
    plt.show()

        
    print(hist.bins)
    print(hist.F)
    print(hist.F[:, A].sum())
        

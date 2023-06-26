## ddffd
import scipy as sp
import numpy as np
import pylab as plt
import time

## Adjust Path
import sys
import pathlib

p = pathlib.Path('~/src/casino').expanduser()
print(p)
sys.path.append(str(p))
print(sys.path)

import casino as cs

fig = plt.figure()
ax = fig.gca()

## sdsdf

data  = np.random.normal(scale=100, size=1000000)
nBins = 4*4


## ddffd
tic = time.time()
dist = cs.ECHAP(nBins)
dist.fit(data.copy())
toc = time.time()

print(f'ECHAP took {toc - tic} seconds.')

## Create the bins
bl = dist.bl
bu = dist.bu
f  = dist.f
ff = f.cumsum()
ff = ff/ff[-1]
ax.plot([bl[0]] + list(bu), [0]+list(ff),linewidth=2 , label='ECHAP',color='blue')

## Calculate the cum distribution funtion for the histogram.
minD = data.min()
maxD = data.max()
xx = np.linspace(minD, maxD, 100)
yy = sp.stats.norm(scale=100).cdf(xx)
ax.plot(xx, yy ,linewidth=2 , label='Actual', color='green', linestyle='--')
## dfd




tic = time.time()
dist = cs.SECHAP(nBins)
dist.fit(data.copy())
toc = time.time()
print(f'SECHAP took {toc - tic} seconds.')

bl = dist.bl
bu = dist.bu
f  = dist.fa
ff = f.cumsum()
ff = ff/ff[-1]
ax.plot([bl[0]] + list(bu), [0]+list(ff),linewidth=2 , label='SECHAP',color='red')






plt.show()












## print
for ll, uu, ff in zip(bl, bu, f):
    print(f'[{ll:.0f}, {uu:.0f}) -> {ff:.0f}', end=' | ')

print()

## plot
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.gca()

# w = bins[1:] - bins[:-1]

# print(w)
# ax.bar(bl, f, width=bu-bl, align='edge')
# plt.show()
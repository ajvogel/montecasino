## ddffd
import scipy as sp
import numpy as np
import pylab as plt

## Adjust Path
import sys
import pathlib

p = pathlib.Path('~/src/casino').expanduser()
print(p)
sys.path.append(str(p))
print(sys.path)

import casino as cs

## sdsdf

data  = np.random.normal(scale=100, size=1000000)
nBins = 4*4


## ddffd

dist = cs.ECHAP(nBins)
dist.fit(data.copy())

## Create the bins
bl = dist.bl
bu = dist.bu
f  = dist.f


## Calculate the cum distribution funtion for the histogram.
minD = data.min()
maxD = data.max()
xx = np.linspace(minD, maxD, 100)
yy = sp.stats.norm(scale=100).cdf(xx)

## dfd

ff = f.cumsum()
ff = ff/ff[-1]
fig = plt.figure()
ax = fig.gca()
print(bl[0],bu[0], ff[0])
ax.plot([bl[0]] + list(bu), [0]+list(ff),linewidth=2 , label='ECHAP',color='blue')
ax.plot(xx, yy ,linewidth=2 , label='Actual', color='green', linestyle='--')
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
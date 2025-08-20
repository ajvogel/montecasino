import sys
import pathlib
import time
import casino as cs
#from casino.random import normal

#print(normal(100, 10))

print('Running...')
tic = time.time()

out = cs.SUM(1000, cs.RandInt(1,6)).compute()
# tri1 = cs.Triangular(10, 30, 50)

# bla = tri1


# a = cs.Triangular(10, 30, 50) + cs.Triangular(10, 30, 50)



# for i in range(50):

#     bla = bla + cs.Triangular(10, 30, 50)
#     # print(bla.bins)
#     # print(bla.cnts)

toc = time.time()

print(f'Took {(toc - tic) * 1000} milliseconds')


# print(bla.lower(), bla.upper())
# print(bla.centroids())
# print(bla.weights())


# import pylab as plt
# import numpy as np


# fig = plt.figure()
# ax = fig.gca()


# x = np.arange(bla.lower(), bla.upper())
# print(x)
# y = np.array([bla.pmf(xx) for xx in x])


# ax.vlines(x, 0, y, color='blue', zorder=0)
# ax.scatter(bla.centroids(), np.zeros_like(bla.centroids()), color='red', zorder=1)
# plt.tight_layout()
# plt.show()

# cnts = bla.getCountArray()
# lwr  = bla.getLowerArray()
# upr  = bla.getUpperArray()

# for l, u, c in zip(lwr, upr, cnts):
#     print(f'[{l: >8d}; {u: >8d}): {u - l:>5d}: {c/cnts.sum():e}')

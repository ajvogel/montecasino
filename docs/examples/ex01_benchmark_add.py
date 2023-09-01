import sys
import pathlib
import time
import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Triangular(10, 30, 50)

bla = tri1 

for i in range(250):
    bla = bla + cs.Triangular(10, 30, 50)

toc = time.time()

print(f'Took {(toc - tic) * 1000} milliseconds')


cnts = bla.getCountArray()
lwr  = bla.getLowerArray()
upr  = bla.getUpperArray()

for l, u, c in zip(lwr, upr, cnts):
    print(f'[{l: >8d}; {u: >8d}): {u - l:>5d}: {c/cnts.sum():e}')

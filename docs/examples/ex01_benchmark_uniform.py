import sys
import pathlib
import time
import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Uniform(1,6)
N = 150

bla = tri1 

for i in range(N - 1):
    bla = bla + cs.Uniform(1,6)

toc = time.time()

print(f'Took {(toc - tic) * 1000} milliseconds')


cnts = bla.getCountArray()
lwr  = bla.getLowerArray()
upr  = bla.getUpperArray()

for l, u, c in zip(lwr, upr, cnts):
    print(f'[{l: >8d}; {u: >8d}): {u - l:>5d}: {c/cnts.sum():e}')


print(f' {N*1} -> {6*N}')

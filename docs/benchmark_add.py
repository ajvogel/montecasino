import sys
import pathlib
import time

from numpy import tri
sys.path.append(str((pathlib.Path(__file__) / '..' / '..').resolve()))

import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Triangular(10, 30, 50)

bla = tri1

for i in range(15):
    bla = bla + cs.Triangular(10, 30, 50)

toc = time.time()

# print(len(bla.w))


# cs.printPMF(bla)

print(f'Took {(toc - tic) * 1000} milliseconds')




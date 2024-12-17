import sys
import pathlib
import time

from numpy import tri
sys.path.append(str((pathlib.Path(__file__) / '..' / '..').resolve()))

import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Uniform(1, 100)

bla = tri1

for i in range(50):
    bla = bla + cs.Uniform(1, 100)

toc = time.time()

# print(len(bla.w))


# cs.printPMF(bla)

print(f'Took {(toc - tic) } seconds...')




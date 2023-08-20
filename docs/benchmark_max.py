import sys
import pathlib
import time

import numpy as np
sys.path.append(str((pathlib.Path(__file__) / '..' / '..').resolve()))

import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Triangular(10, 30, 50, endpoints=True)

tris = [tri1]*5

# out = tri1

out = cs.max(*tris)


toc = time.time()


print('Done. Printing')

cs.printPMF(out)


print(f'Took {(toc - tic) * 1000} milliseconds')




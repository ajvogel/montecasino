import sys
import pathlib
import time

import numpy as np
sys.path.append(str((pathlib.Path(__file__) / '..' / '..').resolve()))

import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Triangular(10, 50, 30, endpoints=True)

tris = [tri1]*100

out = cs.max(*tris)

# for i in range(5):
#     out = cs.max(out,cs.Triangular(10, 50, 30))


toc = time.time()


print('Done. Printing')

cs.printPMF(out)


A = np.array([out.k, out.w])
print(A.T)
print(A.shape)

print(f'Took {(toc - tic) * 1000} milliseconds')




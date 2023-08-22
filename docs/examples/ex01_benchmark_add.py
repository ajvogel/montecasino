import sys
import pathlib
import time
import casino as cs

print('Running...')
tic = time.time()
tri1 = cs.Triangular(10, 30, 50)

bla = tri1 

for i in range(50):
    bla = bla + cs.Triangular(10, 30, 50)

toc = time.time()

print(f'Took {(toc - tic) * 1000} milliseconds')
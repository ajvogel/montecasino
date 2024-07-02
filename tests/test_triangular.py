import casino as cs
import numpy as np


def test_triangular_connectivity():
    tri1 = cs.Triangular(10, 30, 50)
    bla = tri1 + tri1 + tri1
    print(bla)

#     bla._assertConnected()

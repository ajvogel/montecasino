import casino as cs
import numpy as np

def test_isCompiled():
    rv = cs.Digest()
    rv._assertCompiled()

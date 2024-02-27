import casino as cs
import numpy as np

def test_isCompiled():
    rv = cs.RandomVariable()
    rv._assertCompiled()

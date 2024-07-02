import casino as cs
import numpy as np




def test_findLastLesserOrEqualIndex():
    hist = cs.RandomVariable()

    #             0   1   2   3   4    5   6   7   8
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0., 0.])
    hist.bins    = x
    hist.nActive = 6

    assert hist._findLastLesserOrEqualIndex(3)  == 1
    assert hist._findLastLesserOrEqualIndex(5)  == 2
    assert hist._findLastLesserOrEqualIndex(9)  == 4    
    assert hist._findLastLesserOrEqualIndex(2)  == 1
    assert hist._findLastLesserOrEqualIndex(11) == 5

    x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    hist.bins    = x
    hist.nActive = 0
    
    assert hist._findLastLesserOrEqualIndex(5) == -1


def test_shiftRightAndInsert():
    hist = cs.RandomVariable(maxBins=8)

    #             0   1   2   3   4    5    6    7   8
    x = np.array([0., 2., 4., 6., 8., 10., 12., 14., 0.])
    y = np.array([1., 1., 1., 1., 1.,  1.,  1.,  1., 0.])
    hist.bins    = x
    hist.cnts    = y
    hist.nActive = 8

    hist._shiftRightAndInsert(2, 5, 1)

    print(hist.bins)
    print(hist.cnts)

    np.testing.assert_array_equal(hist.bins, np.array([0., 2., 4., 5., 6., 8., 10., 12., 14.]))

    
    hist = cs.RandomVariable(maxBins=7)

    #             0   1   2   3   4    5   6   7 
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.bins    = x
    hist.cnts    = y
    hist.nActive = 6

    hist._shiftRightAndInsert(2, 5, 1)

    print(hist.bins)
    print(hist.cnts)

    np.testing.assert_array_equal(hist.bins, np.array([0., 2., 4., 5., 6., 8., 10., 0.]))

    #             0   1   2   3   4    5   6   7 
    x = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.bins    = x
    hist.cnts    = y
    hist.nActive = 0

    hist._shiftRightAndInsert(-1, 5, 1)

    print(hist.bins)
    print(hist.cnts)

    np.testing.assert_array_equal(hist.bins, np.array([5., 0., 0., 0., 0., 0., 0., 0.]))

def test_LeftAndOverride():
    hist = cs.RandomVariable(maxBins=7)

    #             0   1   2   3   4    5   6   7 
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.bins    = x
    hist.cnts    = y
    hist.nActive = 6

    hist._shiftLeftAndOverride(2)

    np.testing.assert_array_equal(hist.bins, np.array([0., 2., 6., 8., 10., 0., 0., 0.]))
    
    hist = cs.RandomVariable(maxBins=7)

    #             0   1   2   3   4    5   6   7 
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.bins    = x
    hist.cnts    = y
    hist.nActive = 6

    hist._shiftLeftAndOverride(4)

    np.testing.assert_array_equal(hist.bins, np.array([0., 2., 4., 6., 10., 0., 0., 0.]))

    hist = cs.RandomVariable(maxBins=6)

    #             0   1   2   3   4   5   6 
    x = np.array([1., 2., 3., 4., 5., 6., 7.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0.])
    hist.bins    = x
    hist.cnts    = y
    hist.nActive = 7

    hist._shiftLeftAndOverride(4)

    np.testing.assert_array_equal(hist.bins, np.array([1., 2., 3., 4., 6., 7., 0.]))




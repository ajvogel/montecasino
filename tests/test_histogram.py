import casino as cs
import numpy as np




def test_findLastLesserOrEqualIndex():
    hist = cs.TDigest()

    #             0   1   2   3   4    5   6   7   8
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0., 0.])
    # hist.bins    = x
    hist.setBins(x)
    hist.setActiveBinCount(6)


    assert hist._findLastLesserOrEqualIndex(3)  == 1
    assert hist._findLastLesserOrEqualIndex(5)  == 2
    assert hist._findLastLesserOrEqualIndex(9)  == 4
    assert hist._findLastLesserOrEqualIndex(2)  == 1
    assert hist._findLastLesserOrEqualIndex(11) == 5

    x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    hist.setBins(x)
    hist.setActiveBinCount(0)
    # hist.bins    = x
    # hist.nActive = 0

    assert hist._findLastLesserOrEqualIndex(5) == -1


def test_shiftRightAndInsert():
    hist = cs.TDigest(maxBins=8)

    #             0   1   2   3   4    5    6    7   8
    x = np.array([0., 2., 4., 6., 8., 10., 12., 14., 0.])
    y = np.array([1., 1., 1., 1., 1.,  1.,  1.,  1., 0.])
    hist.setBins(x)
    hist.setWeights(y)
    hist.setActiveBinCount(8)



    hist._shiftRightAndInsert(2, 5, 1)



    np.testing.assert_array_equal(hist.getBins(), np.array([0., 2., 4., 5., 6., 8., 10., 12., 14.]))


    hist = cs.TDigest(maxBins=7)

    #             0   1   2   3   4    5   6   7
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.setBins(x)
    hist.setWeights(y)
    hist.setActiveBinCount(6)


    hist._shiftRightAndInsert(2, 5, 1)



    np.testing.assert_array_equal(hist.getBins(), np.array([0., 2., 4., 5., 6., 8., 10., 0.]))

    #             0   1   2   3   4    5   6   7
    x = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.setBins(x)
    hist.setWeights(y)
    hist.setActiveBinCount(0)

    # hist.nActive = 0

    hist._shiftRightAndInsert(-1, 5, 1)



    np.testing.assert_array_equal(hist.getBins(), np.array([5., 0., 0., 0., 0., 0., 0., 0.]))

def test_LeftAndOverride():
    hist = cs.TDigest(maxBins=7)

    #             0   1   2   3   4    5   6   7
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.setBins(x)
    hist.setWeights(y)
    hist.setActiveBinCount(6)

    # hist.nActive = 6

    hist._shiftLeftAndOverride(2)

    np.testing.assert_array_equal(hist.getBins(), np.array([0., 2., 6., 8., 10., 0., 0., 0.]))

    hist = cs.TDigest(maxBins=7)

    #             0   1   2   3   4    5   6   7
    x = np.array([0., 2., 4., 6., 8., 10., 0., 0.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0., 0.])
    hist.setBins(x)
    hist.setWeights(y)
    hist.setActiveBinCount(6)


    # hist.nActive = 6

    hist._shiftLeftAndOverride(4)

    np.testing.assert_array_equal(hist.getBins(), np.array([0., 2., 4., 6., 10., 0., 0., 0.]))

    hist = cs.TDigest(maxBins=6)

    #             0   1   2   3   4   5   6
    x = np.array([1., 2., 3., 4., 5., 6., 7.])
    y = np.array([0., 0., 0., 0., 0.,  0., 0.])
    hist.setBins(x)
    hist.setWeights(y)
    hist.setActiveBinCount(7)

    # hist.nActive = 7

    hist._shiftLeftAndOverride(4)

    np.testing.assert_array_equal(hist.getBins(), np.array([1., 2., 3., 4., 6., 7., 0.]))

def test_normalApprox_quantile():
    """"""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10000)*std + mu
    x = cs.TDigest(maxBins=64)
    for d in data:
        x.add(d)

    DATA = [
        (-64.485, 0.05),
        (32.551, 0.25),
        (100.00, 0.50),
        (167.449, 0.75),
        (264.485, 0.95)
    ]

    for v, p in DATA:
        dv = abs((v - x.quantile(p)) / v)
        print(f'{v} : {x.quantile(p)} : {dv}')
        assert dv <= 1e-1


def test_normalApprox():
    """"""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10000)*std + mu
    x = cs.TDigest(maxBins=32)
    for d in data:
        x.add(d)


    prob1 = x.cdf(mu + std*1) - x.cdf(mu - std*1)
    prob2 = x.cdf(mu + std*2) - x.cdf(mu - std*2)
    prob3 = x.cdf(mu + std*3) - x.cdf(mu - std*3)

    # We are not using a lot of samples and keeping the accuracy bar low, otherwise
    # running the tests will take to long. In practice increasing the number of
    # sample points will increase the accuracy.

    print(prob1)
    print(prob2)
    print(prob3)

    assert abs(0.6827 - prob1) <= 1e-1
    assert abs(0.9545 - prob2) <= 1e-2
    assert abs(0.9973 - prob3) <= 1e-3

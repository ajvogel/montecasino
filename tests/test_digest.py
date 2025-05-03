import casino as cs
import numpy as np



def test_nActiveCount():
    """Tests that the frequency column adds up to the number of points added"""
    std = 100
    mu  = 100
    N   = 100
    data = np.random.randn(N)*std + mu
    x = cs.Digest(maxBins=16)
    for d in data:
        x.add(d)

    # print(x.nAc-tive)

    assert x.getActiveBinCount() == 16


def test_freqAddsUp():
    """Tests that the frequency column adds up to the number of points added"""
    std = 100
    mu  = 100
    N   = 100
    data = np.random.randn(N)*std + mu
    x = cs.Digest(maxBins=16)
    for d in data:
        x.add(d)

    # print(sum(x.freq))

    assert (sum(x.getWeights()) == N)


def test_normalApprox():
    """"""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = cs.Digest(maxBins=32)
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

def test_normalApprox_quantile():
    """"""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = cs.Digest(maxBins=64)
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

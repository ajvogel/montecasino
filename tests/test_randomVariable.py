import casino as cs
import numpy as np

def test_addTwoUniform():
    """addUniform"""
    x = cs.Uniform(1,6) + cs.Uniform(1,6)
    assert abs(x.pmf(2) - 0.0277) <= 1e-4
    assert abs(x.pmf(3) - 0.0555) <= 1e-4
    assert abs(x.pmf(4) - 0.0833) <= 1e-4
    assert abs(x.pmf(5) - 0.1111) <= 1e-4
    assert abs(x.pmf(6) - 0.1388) <= 1e-4
    assert abs(x.pmf(7) - 0.1666) <= 1e-4
    assert abs(x.pmf(8) - 0.1388) <= 1e-4    
    assert abs(x.pmf(9) - 0.1111) <= 1e-4
    assert abs(x.pmf(10) - 0.0833) <= 1e-4    
    assert abs(x.pmf(11) - 0.0555) <= 1e-4
    assert abs(x.pmf(12) - 0.0277) <= 1e-4    

# def test_nActiveCount():
#     """Tests that the frequency column adds up to the number of points added"""
#     std = 100
#     mu  = 100
#     N   = 100
#     data = np.random.randn(N)*std + mu    
#     x = cs.RandomVariable(maxBins=16)
#     for d in data:
#         x.add(d)

#     print(x.nActive)

#     assert x.nActive == 16


def test_binConnectivity():
    """Checks that bins don't become disconnected."""
    np.random.seed(300)
    std = 100
    mu  = 1000
    N   = 100
    data = np.random.randn(N)*std + mu    
    x = cs.RandomVariable(maxBins=16)
    for e, d in enumerate(data):
        print()
        print(f'Adding {d} as point {e+1}...')
        x.add(d)

    x._assertConnected()

def test_freqAddsUp():
    """Tests that the frequency column adds up to the number of points added"""
    std = 100
    mu  = 100
    N   = 100
    data = np.random.randn(N)*std + mu    
    x = cs.RandomVariable(maxBins=16)
    for d in data:
        x.add(d)

    print(sum(x.freq))

    assert (sum(x.freq) == N)
    

def test_normalApprox():
    """"""
    std = 100
    mu  = 100
    data = np.random.randn(1000)*std + mu
    x = cs.RandomVariable(maxBins=32)
    for d in data:
        x.add(d)


    prob = x.cdf(mu + std*2) - x.cdf(mu - std*2)

    print(x.lower)
    print(x.upper)
    print(x.freq)

    print(prob)
    print(abs(0.95 - prob))

    assert abs(0.95 - prob) <= 5e-3


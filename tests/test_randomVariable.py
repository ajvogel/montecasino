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

def test_nActiveCount():
    """Tests that the frequency column adds up to the number of points added"""
    std = 100
    mu  = 100
    N   = 100
    data = np.random.randn(N)*std + mu    
    x = cs.RandomVariable(maxBins=16)
    for d in data:
        x.add(d)

    # print(x.nAc-tive)           

    assert x.activeBins() == 16


def test_binConnectivity():
    """Checks that bins don't become disconnected."""
    np.random.seed(300)
    std = 100
    mu  = 1000
    N   = 100
    data = np.random.randn(N)*std + mu    
    x = cs.RandomVariable(maxBins=16)
    for e, d in enumerate(data):
        # print()
        print(f'Adding {d} as point {e+1}...')
        x.add(d)
        if e >= 16:
            x._assertConnected()
            # assert False        

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

    # print(sum(x.freq))

    assert (sum(x.getFrequencies()) == N)
    

def test_normalApprox():
    """"""
    std = 100
    mu  = 100
    data = np.random.randn(1000)*std + mu
    x = cs.RandomVariable(maxBins=32)
    for d in data:
        x.add(d)


    prob1 = x.cdf(mu + std*1) - x.cdf(mu - std*1)
    prob2 = x.cdf(mu + std*2) - x.cdf(mu - std*2)
    prob3 = x.cdf(mu + std*3) - x.cdf(mu - std*3)

    # We are not using a lot of samples and keeping the accuracy bar low, otherwise
    # running the tests will take to long. In practice increasing the number of 
    # sample points will increase the accuracy.

    assert abs(0.6827 - prob1) <= 1e-1
    assert abs(0.9545 - prob2) <= 1e-2
    assert abs(0.9973 - prob3) <= 1e-3  


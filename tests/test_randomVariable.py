import casino as cs

def test_addTwoUniform():
    """addUniform"""
    x = cs.Uniform(1,6) + cs.Uniform(1,6)
    print(x.lower)
    print(x.upper)
    print(x.freq)
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


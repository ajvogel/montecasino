import casino as cs
import numpy as np

# def test_addTwoUniform():
#     """addUniform"""
#     x = cs.Uniform(1,6) + cs.Uniform(1,6)
#     # print(x.lower)
#     # print(x.upper)
#     # print(x.count)
#     # print(x.known)
#     # assert x._assertConnected()

#     # print(x.nActive)
#     # print(x.maxBins)

#     for i in range(2, 13):
#         print(i, x.pmf(i))

#     print("Bins = ",x.getBins())
#     print("Wgts = ",x.getWeights())

#     assert abs(x.pmf(2) - 0.0277) <= 1e-4
#     assert abs(x.pmf(3) - 0.0555) <= 1e-4
#     assert abs(x.pmf(4) - 0.0833) <= 1e-4
#     assert abs(x.pmf(5) - 0.1111) <= 1e-4
#     assert abs(x.pmf(6) - 0.1388) <= 1e-4
#     assert abs(x.pmf(7) - 0.1666) <= 1e-4
#     assert abs(x.pmf(8) - 0.1388) <= 1e-4
#     assert abs(x.pmf(9) - 0.1111) <= 1e-4
#     assert abs(x.pmf(10) - 0.0833) <= 1e-4
#     assert abs(x.pmf(11) - 0.0555) <= 1e-4
#     assert abs(x.pmf(12) - 0.0277) <= 1e-4


# def test_lowerBound_and_upperBound():
#     x = cs.Uniform(1,6) + cs.Uniform(1,6)
#     assert x.lower() == 2
#     assert x.upper() == 12

# def test_nActiveCount():
#     """Tests that the frequency column adds up to the number of points added"""
#     std = 100
#     mu  = 100
#     N   = 100
#     data = np.random.randn(N)*std + mu
#     x = cs.RandomVariable(maxBins=16)
#     for d in data:
#         x.add(d)

#     # print(x.nAc-tive)

#     assert x.getActiveBinCount() == 16

# NO LONGER APPLICABLE
# def test_binConnectivity():
#     """Checks that bins don't become disconnected."""
#     np.random.seed(300)
#     std = 100
#     mu  = 1000
#     N   = 100
#     data = np.random.randn(N)*std + mu
#     x = cs.RandomVariable(maxBins=16)
#     for e, d in enumerate(data):
#         # print()
#         print(f'Adding {d} as point {e+1}...')
#         x.add(d)
#         if e >= 16:
#             x._assertConnected()
#             # assert False

#     x._assertConnected()

# def test_freqAddsUp():
#     """Tests that the frequency column adds up to the number of points added"""
#     std = 100
#     mu  = 100
#     N   = 100
#     data = np.random.randn(N)*std + mu
#     x = cs.RandomVariable(maxBins=16)
#     for d in data:
#         x.add(d)

    # print(sum(x.freq))

    # assert (sum(x.getWeights()) == N)


# def test_normalApprox():
#     """"""
#     std = 100
#     mu  = 100
#     np.random.seed(31337)
#     data = np.random.randn(10000)*std + mu
#     x = cs.RandomVariable(maxBins=32)
#     for d in data:
#         x.add(d)


#     prob1 = x.cdf(mu + std*1) - x.cdf(mu - std*1)
#     prob2 = x.cdf(mu + std*2) - x.cdf(mu - std*2)
#     prob3 = x.cdf(mu + std*3) - x.cdf(mu - std*3)

#     # We are not using a lot of samples and keeping the accuracy bar low, otherwise
#     # running the tests will take to long. In practice increasing the number of
#     # sample points will increase the accuracy.

#     print(prob1)
#     print(prob2)
#     print(prob3)

#     assert abs(0.6827 - prob1) <= 1e-1
#     assert abs(0.9545 - prob2) <= 1e-2
#     assert abs(0.9973 - prob3) <= 1e-3

# def test_normalApprox_quantile():
#     """"""
#     std = 100
#     mu  = 100
#     np.random.seed(31337)
#     data = np.random.randn(10000)*std + mu
#     x = cs.RandomVariable(maxBins=64)
#     for d in data:
#         x.add(d)

#     DATA = [
#         (-64.485, 0.05),
#         (32.551, 0.25),
#         (100.00, 0.50),
#         (167.449, 0.75),
#         (264.485, 0.95)
#     ]

#     for v, p in DATA:
#         dv = abs((v - x.quantile(p)) / v)
#         print(f'{v} : {x.quantile(p)} : {dv}')
#         assert dv <= 1e-1

    #assert False




def test_randInt_sample():
    dice = cs.RandInt(1,6).sample()
    assert 1 <= dice <= 6

# def test_randInt_compute():
#     dice = cs.RandInt(1,6).compute()

#     assert dice.getActiveBinCount() == 6
#     print(dice)
#     print(dice.getBins())
#     print(dice.getWeights())

#     assert False

def test_randInt_distribution():
    dice = cs.RandInt(1,6).compute()
    nActive = dice.getActiveBinCount()
    minW = min(dice.getWeights()[:nActive])
    maxW = max(dice.getWeights()[:nActive])

    avgW = (maxW + minW) / 2

    assert nActive == 6

    print(dice.getWeights())

    print(minW)
    print(maxW)
    print(maxW - minW)
    print((maxW - minW) / avgW)

    assert abs(maxW - minW) / avgW <= 1e-1




def test_sumDices():
    data6 = [
        [6,0.00214334705075],
        [7,0.0128600823045],
        [8,0.0450102880658],
        [9,0.120027434842],
        [10,0.270061728395],
        [11,0.54012345679],
        [12,0.977366255144],
        [13,1.62037037037],
        [14,2.48842592593],
        [15,3.57081618656],
        [16,4.81610082305],
        [17,6.12139917695],
        [18,7.35382373114],
        [19,8.37191358025],
        [20,9.04706790123],
        [21,9.28497942387],
        [22,9.04706790123],
        [23,8.37191358025],
        [24,7.35382373114],
        [25,6.12139917695],
        [26,4.81610082305],
        [27,3.57081618656],
        [28,2.48842592593],
        [29,1.62037037037],
        [30,0.977366255144],
        [31,0.54012345679],
        [32,0.270061728395],
        [33,0.120027434842],
        [34,0.0450102880658],
        [35,0.0128600823045],
        [36,0.00214334705075]
        ]

    out = [cs.RandInt(1,6) for ii in range(6)]
    out2 = out[0] + out[1] + out[2] + out[3] + out[4] + out[5]
    out2 = out2.compute(samples=10_000)
    #out2 = cs.SUM(5, cs.RandInt(1,6)).compute()

    # cnts = out2.getCountArray()
    # print(sum(cnts))
    #
    yActual = np.array([d[1]/100 for d in data6]).cumsum()
    yTest   = np.array([out2.cdf(k) for k in range(6,37)])

    with open('output.csv','w') as fout:
        fout.write('Actual,Estimate\n')
        for a, e in zip(yActual, yTest):
            fout.write(f'{a*100:3.3f},{e*100:3.3f}\n')

    error   =  (((yTest - yActual)**2).mean())**0.5

    assert error < 5.e-2
    print(yActual)
    print(yTest)
    print(error)

    # for k,p in data6:
    #     pp = p/100
    #     err = abs(out2.pmf(k) - pp) / pp

    #     print(f'{k: >2} = {out2.cdf(k):1.9f} <-> {pp:1.9f} ==> {err:.15e}')


    # assert False

    # for k,p in data6:
    #     pp = p/100
    #     err = abs(out2.pmf(k) - pp)

    #     print(f'{k: >2} = {out2.pmf(k):1.9f} <-> {pp:1.9f} ==> {err:.15e}')


    #     assert (out2.pmf(k) - pp) < 1e-2

    # cnts = out2.getCountArray()
    # lwr  = out2.getLowerArray()
    # upr  = out2.getUpperArray()

    # for l, u, c in zip(lwr, upr, cnts):
    #     print(f'[{l: >8d}; {u: >8d}): {u - l:>5d}: {c/cnts.sum():e}')

    # assert False

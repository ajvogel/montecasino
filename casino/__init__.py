# from .distributions import *
# from . import nbinom

from .core import *
from .nodes import *
from .opcodes import *
import numpy as np

import builtins

# __version__ = '0.2.0.post7.dev0+3e6fe06'

# def max(val1, val2):
#     """
#     Returns the distribution of the max of one or more distributions.
#     """
#     return val1.__max__(val2)
    # val1 = args[0]

    # for val2 in args[1:]:
    #     val1 = val1.__max__(val2)

    # return val1


def max(*args):
    val1 = args[0]

    for val2 in args[1:]:
        val1 = val1.__max__(val2)

    return val1


def min(*args):
    val1 = args[0]

    for val2 in args[1:]:
        val1 = val1.__min__(val2)

    return val1


def printPMF(rv):

    kVec = []
    pVec = []

    cumProb = 0
    k = rv.lowerBound()
    while cumProb < UPPER:
        p = rv.pmf(k)
        kVec.append(k)
        pVec.append(p)
        cumProb += rv.pmf(k)
        k += 1

    maxP = builtins.max(pVec)

    for k, p in zip(kVec, pVec):
        print(f'{k:5} | {"█"*int(p / maxP *100)}')

def fromScipy(rvScipy, maxBins=32, samples=10_000):
    """
    Generates a RandomVariable by repeatably sampling a Scipy frozen distribution
    this allows us to use Scipy and Statsmodels to fit distributions and then
    convert them to Casino Random Variables.
    """
    data = rvScipy.rvs(size=samples)
    return fromArray(data)
    counts = np.ones_like(data)
    rv = RandomVariable(maxBins=maxBins, data=data, counts=counts)
    return rv


def fromArray(array):
    digest = Digest()
    for ar in array:
        digest.add(ar)

    rv = DigestVariable(digest)
    return rv



def plot(digest, nBins=20, lower=None, upper=None, width=0.8, ax=None, *args, **kwds):
    if lower is None:
        lower = digest.lower()

    if upper is None:
        upper = digest.upper()

    bins = np.linspace(lower, upper, nBins)
    y = np.zeros(nBins - 1)
    x = np.zeros(nBins - 1)    

    for i in range(nBins - 1):
        y[i] = digest.cdf(bins[i+1]) - digest.cdf(bins[i])
        x[i] = (bins[i+1] + bins[i])/2

    if ax is None:
        import pylab as plt
        fig = plt.figure()
        ax = fig.gca()

    ax.bar(x, y, width=width*(bins[1] - bins[0]), *args, **kwds)
        

    
    

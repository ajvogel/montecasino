from .distributions import *
from . import nbinom
from .core import *
import builtins

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

def fromScipy(rvScipy, maxBins=32, samples=1000):
    """
    Generates a RandomVariable by repeatably sampling a Scipy frozen distribution
    this allows us to use Scipy and Statsmodels to fit distributions and then
    convert them to Casino Random Variables.
    """
    data = rvScipy.rvs(size=samples)
    counts = np.ones_like(data)
    rv = RandomVariable(maxBins=maxBins, data=data, counts=counts)
    return rv
    

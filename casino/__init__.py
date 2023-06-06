from .discrete import *



def printPMF(rv):
    
    kVec = []
    pVec = []

    cumProb = 0
    k = rv.min()
    while cumProb < UPPER:
        p = rv.pmf(k)
        kVec.append(k)
        pVec.append(p)
        cumProb += rv.pmf(k)
        k += 1 

    maxP = max(pVec)

    for k, p in zip(kVec, pVec):
        print(f'{k:5} | {"█"*int(p / maxP *50)}')
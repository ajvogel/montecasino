import casino as cs
import numpy as np



def test_store_and_load():

    BYTECODE = [
        (cs.OP_PUSH, 314),
        (cs.OP_STORE, 0),
        (cs.OP_PUSH, 3),
        (cs.OP_PUSH, 3),
        (cs.OP_ADD, 0),
        (cs.OP_LOAD, 0)
    ]

    opcodes  = np.array([opco for opco, oper in BYTECODE], dtype=np.double)
    operands = np.array([oper for opco, oper in BYTECODE], dtype=np.double)


    vm = cs.VirtualMachine(opcodes, operands)
    assert vm.sample() == 314


def test_normal_sample():
    std  = 10
    mu = 100

    BYTECODE = [
        (cs.OP_PUSH, mu),
        (cs.OP_PUSH, std),
        (cs.OP_RANDNORM, 0)
    ]

    opcodes  = np.array([opco for opco, oper in BYTECODE], dtype=np.double)
    operands = np.array([oper for opco, oper in BYTECODE], dtype=np.double)

    vm = cs.VirtualMachine(opcodes, operands)
    x = vm.compute()

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

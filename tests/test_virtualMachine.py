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

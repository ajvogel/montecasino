import cython as pyx

@pyx.ccall
def normal() -> pyx.float:
    return snorm()


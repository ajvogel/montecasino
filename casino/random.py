import cython as pyx

@pyx.ccall
def normal(mu: pyx.float , std: pyx.float) -> pyx.float:
    return snorm()*std + mu


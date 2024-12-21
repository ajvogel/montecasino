cdef extern from "rnglib.c":
    float r4_uni_01 ()

cdef extern from "ranlib.c":
    float gennor(float av, float sd)

cpdef float normal(float mu, float std)



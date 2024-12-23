cdef extern from "ranlib.c":
    float snorm()

cpdef float normal(float mu, float std)

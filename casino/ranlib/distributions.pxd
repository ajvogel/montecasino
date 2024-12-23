cdef extern from "rnglib2.c":
    float r4_uni_01 ()
#     void initialize()
#     void set_initial_seed ( int ig1, int ig2 )

cdef extern from "ranlib.c":
    float gennor(float av, float sd)

cpdef float normal(float mu, float std)



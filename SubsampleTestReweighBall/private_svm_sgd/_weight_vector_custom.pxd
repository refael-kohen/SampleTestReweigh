
# WARNING: Do not edit this .pyx file directly, it is generated from its .pyx.tp
cimport numpy as np

cdef class WeightVector64(object):
    cdef readonly double[::1] w
    cdef double *w_data_ptr
    cdef double wscale
    cdef int n_features

    cdef void add_proj_noise(self, double * x_data_ptr, int * x_ind_ptr, int xnnz, double c,
                             int do_proj, double min_proj, double max_proj, int n_features_noise,
                             double * noises) nogil
    cdef void add(self, double *x_data_ptr, int *x_ind_ptr,
                  int xnnz, double c) nogil
    cdef double dot(self, double *x_data_ptr, int *x_ind_ptr,
                    int xnnz) nogil
    cdef void scale(self, double c) nogil
    cdef void reset_wscale(self) nogil
#
# cdef class WeightVector32(object):
#     cdef readonly float[::1] w
#     cdef float *w_data_ptr
#     cdef float wscale
#     cdef int n_features
#
#     cdef void add_proj_noise(self, float * x_data_ptr, int * x_ind_ptr, int xnnz, float c,
#                              int do_proj, float min_proj, float max_proj, int n_features_noise,
#                              float * noises) nogil
#     cdef void add(self, float *x_data_ptr, int *x_ind_ptr,
#                   int xnnz, float c) nogil
#     cdef float dot(self, float *x_data_ptr, int *x_ind_ptr,
#                     int xnnz) nogil
#     cdef void scale(self, float c) nogil
#     cdef void reset_wscale(self) nogil

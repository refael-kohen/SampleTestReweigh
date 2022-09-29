
# cython: binding=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
#         Danny Sullivan <dsullivan7@hotmail.com>
#
# License: BSD 3 clause

# WARNING: Do not edit this .pyx file directly, it is generated from its .pyx.tp

cimport cython
from libc.limits cimport INT_MAX
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

from sklearn.utils._cython_blas cimport _dot, _scal, _axpy

from libc.stdio cimport printf


np.import_array()

cdef class WeightVector64(object):
    """Dense vector represented by a scalar and a numpy array.

    The class provides methods to ``add`` a sparse vector
    and scale the vector.
    Representing a vector explicitly as a scalar times a
    vector allows for efficient scaling operations.

    Attributes
    ----------
    w : ndarray, dtype=double, order='C'
        The numpy array which backs the weight vector.
    w_data_ptr : double*
        A pointer to the data of the numpy array.
    wscale : double
        The scale of the vector.
    n_features : int
        The number of features (= dimensionality of ``w``).
    """

    def __cinit__(self, double[::1] w):

        if w.shape[0] > INT_MAX:
            raise ValueError("More than %d features not supported; got %d."
                             % (INT_MAX, w.shape[0]))
        self.w = w
        self.w_data_ptr = &w[0]
        self.wscale = 1.0
        self.n_features = w.shape[0]


    cdef void add(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                  double c) nogil:
        """Scales sample x by constant c and adds it to the weight vector.

        This operation updates ``sq_norm``.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        """
        cdef int j
        cdef int idx
        cdef double val

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef double* w_data_ptr = self.w_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            w_data_ptr[idx] += val * (c / wscale)

    cdef void add_proj_noise(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                  double c, int do_proj, double min_proj, double max_proj, int n_features_noise, double *noises) nogil:
        """Scales sample x by constant c and adds it to the weight vector.

        This operation updates ``sq_norm``.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        do_proj : int
            flag if do projection or not
        min_proj : double
            min value in projection
        max_proj : double
            max value in projection
        n_features_noise: int
            number of features to add noise (the dimention of x) - if no need to add noise the value is 0
        noises : double*
            The array which holds the noises
        """
        cdef int j
        cdef int idx
        cdef double val
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef double* w_data_ptr = self.w_data_ptr
        cdef double min_proj_scaled = min_proj/wscale
        cdef double max_proj_scaled = max_proj/wscale

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            # printf("%f %lf\n", val, c)
            w_data_ptr[idx] += val * (c / wscale)
            if do_proj and n_features_noise == 0:
                if w_data_ptr[idx] < min_proj_scaled:
                    w_data_ptr[idx] = min_proj_scaled
                elif w_data_ptr[idx] > max_proj_scaled:
                    w_data_ptr[idx] = max_proj_scaled

        for j in range(n_features_noise):
            w_data_ptr[j] += noises[j] / wscale
            if do_proj:
                if w_data_ptr[j] < min_proj_scaled:
                    w_data_ptr[j] = min_proj_scaled
                elif w_data_ptr[j] > max_proj_scaled:
                    w_data_ptr[j] = max_proj_scaled




    cdef double dot(self, double *x_data_ptr, int *x_ind_ptr,
                    int xnnz) nogil:
        """Computes the dot product of a sample x and the weight vector.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x`` (length of x_ind_ptr).

        Returns
        -------
        innerprod : double
            The inner product of ``x`` and ``w``.
        """
        cdef int j
        cdef int idx
        cdef double innerprod = 0.0
        cdef double* w_data_ptr = self.w_data_ptr
        for j in range(xnnz):
            idx = x_ind_ptr[j]
            innerprod += w_data_ptr[idx] * x_data_ptr[j]
        innerprod *= self.wscale
        return innerprod

    cdef void scale(self, double c) nogil:
        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
        small we call ``reset_swcale``."""
        self.wscale *= c

        if self.wscale < 1e-09:
            self.reset_wscale()

    cdef void reset_wscale(self) nogil:
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        _scal(self.n_features, self.wscale, self.w_data_ptr, 1)
        self.wscale = 1.0

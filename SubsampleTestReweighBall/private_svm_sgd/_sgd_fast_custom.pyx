#################################

# Original script is taken from scikit-learn script was written by:

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel (partial_fit support)
#         Rob Zinkov (passive-aggressive)
#         Lars Buitinck
#
# License: BSD 3 clause

# This original script is taken from:
# sklearn\linear_model\_stochastic_gradient.py
# We added to the script a noise in each iteration of the SGD algorithm.

# All changes in the script are signed by the word: # NOISE_CHANGES

import sys
from time import time
from numpy import random
cimport cython
from libc.math cimport exp, log, sqrt, pow, fabs
# NOISE_CHANGES
# Addition of including srand, rand
from libc.stdlib cimport srand, rand, RAND_MAX

# NOISE_CHANGES
# Addition of including malloc, free, parallel, prange, printf
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport printf


# NOISE_CHANGES
# Addition of including srand, rand
from libc.time cimport time as libc_time

# another way to create random number:
from sklearn.utils._random cimport our_rand_r

# Using cupy for random.choice function
# import cupy

# Using numpy for random.choice function
import numpy



cimport numpy as np

from SubsampleTestReweigh.private_svm_sgd._weight_vector_custom cimport WeightVector64 as WeightVector
from sklearn.utils._seq_dataset cimport SequentialDataset64 as SequentialDataset

np.import_array()


cdef class LossFunction:
    """Base class for convex loss functions"""

    cdef double loss(self, double p, double y, double c, double intercept) nogil:
        """Evaluate the loss function.

        Parameters
        ----------
        p : double
            The prediction, `p = w^T x + intercept`.
        y : double
            The true value (aka target).
        c: double
            1/alpha
        intercept: double 
            The intercept value

        Returns
        -------
        double
            The loss evaluated at `p` and `y`.
        """
        return 0.

    def py_dloss(self, double p, double y, double c, double intercept):
        """Python version of `dloss` for testing.

        Pytest needs a python function and can't use cdef functions.

        Parameters
        ----------
        p : double
            The prediction, `p = w^T x`.
        y : double
            The true value (aka target).
        c: double
            1/alpha
        intercept: double
            The intercept value

        Returns
        -------
        double
            The derivative of the loss function with regards to `p`.
        """
        return self.dloss(p, y, c, intercept)

    def py_loss(self, double p, double y, double c, double intercept):
        """Python version of `loss` for testing.

        Pytest needs a python function and can't use cdef functions.

        Parameters
        ----------
        p : double
            The prediction, `p = w^T x + intercept`.
        y : double
            The true value (aka target).
        intercept: double
            The intercept value

        Returns
        -------
        double
            The loss evaluated at `p` and `y`.
        """
        return self.loss(p, y, c, intercept)

    cdef double dloss(self, double p, double y, double c, double intercept) nogil:
        """Evaluate the derivative of the loss function with respect to
        the prediction `p`.

        Parameters
        ----------
        p : double
            The prediction, `p = w^T x`.
        y : double
            The true value (aka target).
        c: double
            1/alpha
        intercept: double 
            The intercept value

        Returns
        -------
        double
            The derivative of the loss function with regards to `p`.
        """
        return 0.

cdef class Classification(LossFunction):
    """Base class for loss functions for classification"""

    cdef double loss(self, double p, double y, double c, double intercept) nogil:
        return 0.

    cdef double dloss(self, double p, double y, double c, double intercept) nogil:
        return 0.

cdef class CircleHinge(Classification):
    """CircleHinge loss for binary classification tasks with y in {-1,1}

    All inputs are of form ||x||^2 and w is hyperplace. All dots that holds <||x||^2, w> > intercept are positive
    otherelse are negative.

    It is equivalent to problem that all x inside circle with radius of "intercept" are negative and outside are
    positive.

    The loss function is <||x||^2, w> - intercept > 0 indicate positive classification (in contrast +intercept in
    ragular hinge loss where the hyperplane classify all examples above (- intercept) as positive.

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    """

    cdef double threshold

    def __init__(self, double threshold=0.0):
        self.threshold = threshold

    cdef double loss(self, double p, double y, double c, double intercept) nogil:
        # the intercept is positive number but the additional coordinate of the sample is -1 (so we preserve the
        # hypothesis in range of [0,1]
        cdef double z = (p - intercept) * c * y
        if z <= self.threshold:
            # return minus z because the loss need to be positive number (because z is negative under this condition)
            return self.threshold - z
        return 0.0

    cdef double dloss(self, double p, double y, double c, double intercept) nogil:
        cdef double z = (p - intercept) * c * y
        if z <= self.threshold:
            return -y*c # the loss is minus z, so the gradient is minus of y*x*c (the x is calculated outside of this function)
        return 0.0

    def __reduce__(self):
        return CircleHinge, (self.threshold,)

@cython.boundscheck(False)
@cython.wraparound(False)
def _plain_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
               double intercept,
               int average,
               np.ndarray[double, ndim=1, mode='c'] average_weights,
               double average_intercept,
               LossFunction loss_f,
               double alpha,
               SequentialDataset dataset,
               int max_iter,
               int fit_intercept,
               np.uint32_t seed,
               # NOISE_CHANGES - eta0 is D/G in non-private, or D/sigma in private
               # where D is diameter of the hypotheses class and G is the upper bound on the 2-norm of the gradient
               double eta0,

               # NOISE_CHANGES - new parameters
               int random_blocks_num, # divide the maximum iterations number to random_clocks_num and create blocks of random numbers (less memory usage of choice and rundom functions) 
               int eta_start_iter,  # from which iteration to start the eta (eta divided by sqrt(iter) - we can start to divide in big iter other than 1 if the updates in the first iterations are meaningless)
               np.ndarray[str, ndim=1, mode='c'] logs,
               bint early_stopping,
               double early_stopping_score,
               int early_stopping_n_iter,
               np.ndarray[double, ndim=1, mode='c'] sample_dist,
               int batch_size=1,
               int do_proj=1,
               double min_proj=0,
               double max_proj=1,
               int private=0,
               double mean_noise=0,
               double sigma_noise=0,
               int noise_seed=0,
               int verbose=10, # original variable - print something each verbose iterations (odd number - print also debug output)
               ):
    """SGD for generic loss functions and penalties with optional averaging

    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated vector of weights.
    intercept : double
        The initial intercept.
    average : int
        The number of iterations before averaging starts. average=1 is
        equivalent to averaging for all iterations. average=0 is not average at all.
    average_weights : ndarray[double, ndim=1]
        The average weights as computed for ASGD. Should be None if average
        is 0.
    average_intercept : double
        The average intercept for ASGD. Should be 0 if average is 0.
    loss_f : LossFunction
        A concrete ``LossFunction`` object.
    alpha : float
        The regularization parameter.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    early_stopping : boolean
        Whether to use a stopping criterion based on the all input learning set.
    early_stopping_score : float
        stop if the score on all learning set is equal or under this score
    early_stopping_n_iter : int
        check the score each such iterations
    max_iter : int
        The maximum number of iterations (epochs).
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite. print something each verbose iterations
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    eta0 : double
        The initial learning rate.

    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    average_weights : array shape=[n_features]
        The averaged weights across iterations. Values are valid only if
        average > 0.
    average_intercept : float
        The averaged intercept across iterations.
        Values are valid only if average > 0.
    n_iter_ : int
        The actual number of iter (epochs).
    """
    t_start = time()

    # get the data information into easy vars
    cdef Py_ssize_t n_samples = dataset.n_samples
    cdef Py_ssize_t n_features = weights.shape[0]

    cdef WeightVector w = WeightVector(weights)
    cdef double* w_ptr = &weights[0]
    cdef WeightVector aw
    if average > 0:
        aw = WeightVector(average_weights)
    cdef double *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef double* ps_ptr = NULL

    cdef int xnnz
    cdef double p = 0.0
    cdef double update = 0.0
    cdef double update_intercept = 0.0
    cdef double intercept_update = 0.0
    cdef double sumlossbatch = 0.0
    cdef double y = 0.0
    cdef double sample_weight
    cdef double loss = 0.0
    cdef double dloss = 0.0
    cdef double w_loss = 0.0
    cdef double weighted_loss = 0.0
    cdef double one_labeled_loss = 0
    cdef double zero_labeled_loss = 0


    cdef int iter_num = 0
    cdef int example_index = 0
    cdef int t = eta_start_iter
    cdef int logs_idx = 0


    # NOISE_CHANGES
    cdef np.ndarray[np.double_t, ndim=1] avg_x_dloss_batch_numpy = numpy.zeros(n_features)
    cdef np.ndarray[int, ndim=1] avg_x_dloss_batch_idx = numpy.arange(n_features, dtype=numpy.int32)
    cdef WeightVector avg_x_dloss_batch = WeightVector(avg_x_dloss_batch_numpy)
    cdef int j = 0
    cdef int jj = 0 # for validation loop on all examples
    cdef double avg_x_dloss_intercept = 0
    # Indexes of examples in each batch
    cdef np.ndarray[int, ndim=1] batch_indexes
    cdef int* batch_indexes_ptr = NULL
    cdef np.ndarray[np.double_t, ndim=1] noises
    cdef np.ndarray[int, ndim=1] x_ind_dense = numpy.arange(n_features, dtype=numpy.int32)
    cdef int* x_ind_ptr_dense = &x_ind_dense[0]
    cdef double tmpo
    cdef double *tmp = NULL
    current_random_block = -1
    random_block_sizes = [int(max_iter / random_blocks_num)]*(random_blocks_num-1) + [max_iter - int(max_iter / random_blocks_num)*(random_blocks_num-1)]
    # NOISE_CHANGES - set the randomizing of normal distribution with seed or time
    if noise_seed!=0:
        numpy.random.seed(noise_seed)
    else:
        numpy.random.seed(None)

    with nogil:
        if verbose:
            with gil:
                if private:
                    logs[logs_idx] = "-- SGD - Run private version of SGDClassifier, with Gaussian noise ~ N({}, {})".format(mean_noise, sigma_noise)
                    logs_idx += 1
                    # printf("-- Run private version of SGDClassifier, with Gaussian noise ~ N(%g, %g)\n", mean_noise, sigma_noise)
                else:
                    logs[logs_idx] = "-- SGD - Run custom version of SGDClassifier"
                    logs_idx += 1
                    # printf("-- Run custom version of SGDClassifier\n")
                logs[logs_idx] = "-- SGD - The total number of iterations is {}, print debug output each {} iterations".format(max_iter, verbose)
                logs_idx += 1
                # printf("-- The total number of iterations is %d, print debug output each %d iterations\n", max_iter, verbose)
        if private:
            with gil:
               #  For efficiency we create noise in advance for all iterations
                noises_all = numpy.random.normal(loc=0, scale=sigma_noise, size=(max_iter, n_features+1))

        # if batch_size == 1:
        #   # For efficiency we choice in advance the examples for all iterations with replace=True
        #   # When batch_size > 1 it is impossible because we need to choice the batch with replace=False
            # with gil:
            #     batch_indexes_all = numpy.random.choice(n_samples, size=(max_iter, 1), replace=True, p=sample_dist).astype(numpy.int32)

        for iter_num in range(max_iter):
            eta = eta0 / sqrt(t)

            # NOISE_CHANGES - The calling to random function added instead of next() function
            # the functions next and random are located in:
            # sklearn\utils\_seq_dataset.pyx
            with gil:
                if batch_size == 1:
                    # For efficiency we choice in advance the examples for all iterations with replace=True
                    # When batch_size > 1 it is impossible because we need to choice the batch with replace=False
                    if current_random_block == -1 or iter_num % random_block_sizes[current_random_block] == 0:
                        current_random_block += 1
                        ss = time()
                        batch_indexes_all = numpy.random.choice(n_samples, size=(random_block_sizes[current_random_block], 1), replace=True,
                                                                p=sample_dist).astype(numpy.int32)
                        print("\t\t\t\t\t\t\tEnd choice: ", (time() - ss) / 60)
                        sys.stdout.flush()
                    batch_indexes = batch_indexes_all[iter_num % random_block_sizes[current_random_block]]
                else:
                    # replace=False, it is not iid. but because the batch is small it is close to iid. we need it for privacy, that in each iteration
                    # the only different example could to be selected only one time.
                    batch_indexes = numpy.random.choice(n_samples, size=batch_size, replace=False, p=sample_dist).astype(numpy.int32)
            batch_indexes_ptr = &batch_indexes[0]

            # NOISE_CHANGES  - start a new batch
            if batch_size > 1:
                avg_x_dloss_batch.scale(0)
                avg_x_dloss_intercept = 0
            # cdef int current_index = our_rand_r(&seed) % n_samples
            # for example_index in batch_indexes:
            sumlossbatch = 0.0
            for j in range(batch_size):
                # sample() takes the index that you send him and update the dataset.current_index
                dataset._sample(&x_data_ptr, &x_ind_ptr, &xnnz, &y, &sample_weight, batch_indexes_ptr[j])
                p = w.dot(x_data_ptr, x_ind_ptr, xnnz)
                loss = loss_f.loss(p, y, 1.0 / alpha, intercept)
                sumlossbatch += (1 if loss > 0 else 0)

                dloss = loss_f.dloss(p, y, 1.0 / alpha, intercept)
                # TODO: NOISE_CHANGES - Verify that eta will be lower than 1
                update = -eta * dloss
                # there is minus because the additional coordinate of the sample is -1 (the intercept is positive
                # because all coordinates of the hypothesis are in range [0,1]
                update_intercept = -update

                # NOISE_CHANGES  - sum the gradients (no part of regularization) in the batch
                if batch_size > 1:
                    avg_x_dloss_batch.add(x_data_ptr, x_ind_ptr, xnnz, dloss)
                    # there is minus because the additional coordinate of the sample is -1 (the intercept is positive
                    # because all coordinates of the hypothesis are in range [0,1]
                    avg_x_dloss_intercept += -dloss
            if verbose > 0 and iter_num % verbose == 0 and iter_num > 0 and batch_size > 1:
                with gil:
                    logs[logs_idx] = "  -- SGD - Iteration number: {}, average batch loss: {}".format(iter_num + 1, sumlossbatch/(batch_size))
                    logs_idx += 1
                    if verbose % 2 != 0:
                        print("  -- SGD - Iteration number: %d, average batch loss: %f" % (iter_num + 1, sumlossbatch/(batch_size)))

            if batch_size > 1:
                avg_x_dloss_batch.scale(1.0 / batch_size)
                avg_x_dloss_intercept /= batch_size
                x_data_ptr = &avg_x_dloss_batch_numpy[0]
                x_ind_ptr = x_ind_ptr_dense
                xnnz = n_features
                update = -eta
                update_intercept = avg_x_dloss_intercept*update

            if private:
                with gil:
                    noises = -eta*noises_all[iter_num]
                # noises = -eta*numpy.random.normal(loc=0, scale=sigma_noise, size=n_features+1)

            n_features_noise = n_features if private else 0
            if update != 0.0 or private:
                w.add_proj_noise(x_data_ptr, x_ind_ptr, xnnz, update,
                                 do_proj, min_proj, max_proj, n_features_noise, &noises[0])
                if fit_intercept == 1:
                    intercept += update_intercept
                    if private:
                        intercept += noises[-1]
                    if do_proj:
                        if intercept < min_proj:
                            intercept = min_proj
                        elif intercept > max_proj:
                            intercept = max_proj

                if 0 < average <= t:
                    # with gil:
                    #     print(*weights, intercept) # for test
                    aw.add(w_ptr, x_ind_ptr_dense, n_features, w.wscale)
                    average_intercept += intercept

            if early_stopping and iter_num > 0 and iter_num % early_stopping_n_iter == 0:# and iter_num > 0:
                weighted_loss = 0.0
                one_labeled_loss = 0.0
                # current index is increased in 1 in each calling next() and updated in each _sample()
                # next() takes the current_index+1, _sample() takes the index that you send him and update the currend_index
                dataset.current_index = -1
                for jj in range(n_samples):
                    dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz, &y, &sample_weight)
                    p = w.dot(x_data_ptr, x_ind_ptr, xnnz)
                    l = loss_f.loss(p, y, 1.0 / alpha, intercept)
                    w_loss = (sample_dist[jj] if l > 0 else 0) # 1 *  the distribution of the sample
                    weighted_loss += w_loss
                    if y==1:
                        one_labeled_loss += w_loss
                    # if y == -1:
                    #     with gil:
                    #         print('0 label:', p, intercept, y)
                    # if l != 0:
                    #     with gil:
                    #         print('worng:', p, intercept, y)
                zero_labeled_loss = weighted_loss - one_labeled_loss
                if weighted_loss <= early_stopping_score:
                    with gil:
                        logs[logs_idx] = "-- SGD - Early stopping after {} iterations - the loss on the all S is: {}, from which 1-labeled-loss: {}, 0-labeled-loss: {}".format(iter_num+1, weighted_loss, one_labeled_loss, zero_labeled_loss)
                        logs_idx += 1
                        logs[logs_idx] = "-- SGD - End run after {} iterations in total training time: {} minutes.".format(
                            iter_num + 1, (time() - t_start) / 60)
                        logs_idx += 1

                    # printf("-- SGD - Early stopping after %d iterations - the loss on the all S is: %f, from which 1-labeled-loss: %f, 0-labeled-loss: %f\n", iter_num+1, weighted_loss, one_labeled_loss, zero_labeled_loss)
                    w.reset_wscale()
                    if average > 0:
                        aw.scale(1.0 / (iter_num - average + 2))
                        aw.reset_wscale()
                    with gil:
                        # return minus of intercept because the predict function do wx+b
                        # scikit-learn-main\scikit-learn-main\sklearn\linear_model
                        return weights, -intercept, average_weights, -average_intercept/(iter_num - average + 2), iter_num+1
                else:
                    with gil:
                        # print(weights, intercept)
                        logs[logs_idx] = "-- SGD - Iteration number: {} - the loss on the all S is: {}, from which 1-labeled-loss: {}, 0-labeled-loss: {}".format(iter_num+1, weighted_loss, one_labeled_loss, zero_labeled_loss)
                        logs_idx += 1
                        # print("-- SGD - Iteration number: {} - the loss on the all S is: {}, from which 1-labeled-loss: {}, 0-labeled-loss: {}\n".format(iter_num+1, weighted_loss, one_labeled_loss, zero_labeled_loss)
            t += 1

        # report epoch information
        # if verbose > 0:
        #     with gil: # print python object - weights
        #         print("-- NNZs: %d, Bias: %.6f, T: %d, Avg. loss: %f"
        #               %(weights.nonzero()[0].shape[0], intercept, iter_num+1, sumlossbatch/batch_size))
        #         print("-- Total training time: %.2f seconds." %(time() - t_start))

    logs[logs_idx] = "-- SGD - End run after {} iterations in total training time: {} minutes.".format(
        iter_num + 1, (time() - t_start) / 60)
    logs_idx += 1

    w.reset_wscale()
    if average > 0:
        aw.scale(1.0 / (iter_num - average + 2))
        aw.reset_wscale()
    # We returns minus of intercept because we suppose thea all coordinates of the hypothesis are in
    # range of [0,1], and therefore the additional coordinate of the example is -1. But predict of scikit learn don't
    # know it and calculate wx+b, therefore we returns -intercept
    # scikit-learn-main\scikit-learn-main\sklearn\linear_model

    # print(average_weights, average_intercept/(iter_num - average + 2), (iter_num - average + 2), iter_num, average, noise_seed) # for test
    return weights, -intercept, average_weights, -average_intercept/(iter_num - average + 2), iter_num + 1
    # return weights, intercept, average_weights, average_intercept, epoch + 1

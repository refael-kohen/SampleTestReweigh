from math import sqrt
from typing import Tuple

import numpy as np
from scipy.stats import norm, chi2

from SubsampleTestReweighBall.utils import *  # import np as numpy or cupy


class Sample:
    def __init__(self, mean: float, std: float, mean_k: float, std_k: float, dim: int, k_coord=None,
                 limit_norm2sq: bool = False, norm2sq: float = float('inf')) -> None:
        """
        :param mean:            expectation
        :param std:             standard deviation
        :param mean_k:          expectation of the k coordinates
        :param std_k:           standard deviation of the k coordinates
        :param dim:             dimension
        :param k_coord:         list of coordinates with other parameters
        :param limit_norm2sq:     limit or not limit the norm2 of the examples
        :param norm2sq:           the value of the norm2sq (if need to limit the norm2sq)
        """
        if k_coord is None:
            k_coord = []
        self.mean = mean
        self.std = std
        self.mean_k = mean_k
        self.std_k = std_k
        self.dim = dim
        self.k_coord = k_coord
        self.limit_norm2sq = limit_norm2sq
        self.norm2sq = norm2sq
        self.mean_arr = np.array([self.mean] * self.dim)
        self.mean_arr[self.k_coord] = self.mean_k
        self.std_arr = np.array([self.std] * self.dim)
        self.std_arr[self.k_coord] = self.std_k

    def get_sample(self, size: int) -> np.ndarray:
        """
        Create sample of examples from the distribution

        :param size: number of examples
        :return: 2d numpy array of (size x self.dim)
        """
        # cupy need to cast size to int
        sample = np.random.normal(loc=self.mean_arr, scale=self.std_arr, size=(int(size), self.dim)) ** 2
        if self.limit_norm2sq:
            sample_norm2sq = np.sum(sample**2, axis=1)
            idx_ge_norm = sample_norm2sq > self.norm2sq
            if idx_ge_norm.any():
                sample[idx_ge_norm] /= np.sqrt(sample_norm2sq[idx_ge_norm].reshape(-1, 1) / self.norm2sq)
        return sample

    @staticmethod
    def get_label(sample: np.ndarray, hypothesis: np.ndarray, intercept: float) -> np.ndarray:
        """
        :param sample:     2d array
        :param hypothesis: 1d array - hypothesis without the intercept
        :param intercept:  the intercept of the hypothesis

        :return: 1d array of 1 and 0.
        """
        # The intercept is positive (in order to all coordinates of the hypothesis will be in the range [0,1], but
        # the additional coordinate of the example is -1, therefore we subtract the intercept
        return np.where((sample @ hypothesis) - intercept > 0, 1, 0)

    def get_true_hypothesis(self, v) -> np.ndarray:
        """
        :return: True hypothesis h of dimension dim+1, with 0 values in all dim coordinates except the k_coord with
             value of 1, and in coordinate d+1 the b threshold value such that cdf(b) is alpha.
        """
        true_hypothesis = np.zeros(v.SampleV.dim + 1)
        true_hypothesis[v.SampleV.k_coord + [v.SampleV.dim]] = 1
        # inverse of cdf - returns the cutoff of x under it the cdf is alpha

        # The intercept is the minus of this value (in get_label function we subtract this value)
        true_hypothesis[v.SampleV.dim] *= v.ModelV.intercept_raw
        return true_hypothesis

    @staticmethod
    def get_pdf(sample_s: np.ndarray, true_hypothesis: np.ndarray, mean_k_t: float, std_k_t: float, mean_k_s: float,
                std_k_s: float, k: int, k_coord: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the pdf of the samples from S according T distribution (for efficiently reason - only on the k
        coordinates, so it can use only for the relation between T and S).

        :param sample_s:            samples from S
        :param true_hypothesis :    true hypothesis on T
        :param mean_k_t:            mean of the k coordinates in T
        :param std_k_t:             std of the k coordinates in T
        :param mean_k_s:            mean of the k coordinates in S
        :param std_k_s:             std of the k coordinates in S
        :param k:                   the k differ coordinates
        :param k_coord:         list of coordinates with other parameters
        :return:                    pdf of the sample of S according to S and T distribution (only on the k coordinates).
        """
        # cannot work with cupy
        # pdf_sample_s_in_T = norm.pdf(x=sample_s.get() @ true_hypothesis, loc=mean_k_t, scale=sqrt(k) * std_k_t)
        # pdf_sample_s = norm.pdf(x=get_np(sample_s) @ get_np(true_hypothesis[:-1]), loc=mean_k_s,
        #                         scale=sqrt(k) * std_k_s)
        # pdf_sample_s_in_T = norm.pdf(x=get_np(sample_s) @ get_np(true_hypothesis[:-1]), loc=mean_k_t,
        #                              scale=sqrt(k) * std_k_t)

        # multiply by true_hypothesis only for efficiency (no need to do np.prod on all elements in the vector -
        # because S and T are equals in these coordinates)

        pdf_sample_t_over_s = []
        # pdf_sample_s_in_T = []
        group_size = 10000
        for i in range(0, sample_s.shape[0], group_size):
            pdf_sample_t_over_s.append(np.prod((std_k_s/std_k_t) * np.exp(-0.5*(sample_s[i:i + group_size, k_coord]**2)*((1/std_k_t**2)-(1/std_k_s**2))), axis=1))
            # pdf_sample_s.append(np.prod(norm.pdf(x=sample_s[i:i + group_size, :], loc=0, scale=1), axis=1))
            # pdf_sample_s_in_T.append(np.prod(norm.pdf(x=sample_s[i:i + group_size, :], loc=0, scale=sqrt(2) * 5), axis=1))

        pdf_sample_t_over_s = np.hstack(pdf_sample_t_over_s)
        # pdf_sample_s = np.hstack(pdf_sample_s)
        # pdf_sample_s_in_T = np.hstack(pdf_sample_s_in_T)
        # fast run but more memory
        # pdf_sample_s = np.prod(norm.pdf(x=sample_s, loc=mean_k_s, scale=std_k_s), axis=1)
        # pdf_sample_s_in_T = np.prod(norm.pdf(x=sample_s, loc=mean_k_t, scale=sqrt(k) * std_k_t), axis=1)


        # pdf_sample_s[pdf_sample_s == 0.0] = np.finfo(np.float64).min
        # pdf_sample_s_in_T[pdf_sample_s_in_T == 0.0] = np.finfo(np.float64).min
        # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', sample_s.shape, (pdf_sample_s/pdf_sample_s_in_T)/np.sum(np.nan_to_num(pdf_sample_s/pdf_sample_s_in_T)))
        # mistake !!! thare are several vectors in with such sum
        # pdf_sample_s = chi2.pdf(x=get_np(sample_s) @ get_np(true_hypothesis[:-1]), df=k)
        # pdf_sample_s_in_T = chi2.pdf(x=get_np(sample_s) @ get_np(true_hypothesis[:-1])/std_k_t**2, df=k)
        # return pdf_sample_s, pdf_sample_s_in_T
        return pdf_sample_t_over_s

# distS = Sample(mean=MEAN_S, std=STD_S, mean_k=mean_k_s, std_k=std_k_s, k_coord=[], dim=4)
# distT = Sample(mean=MEAN_T, std=STD_T, mean_k=mean_k_t, std_k=std_k_t, k_coord=K_COORD, dim=4)
# # print(distS.sample())
# print(distS.get_sample(3))
# print(distT.get_sample(3))

#
# from collections.abc import Sequence
#
# ConnectionOptions = Dict[str, str]
# Address = Tuple[str, int]
# Server = Tuple[Address, ConnectionOptions]
#
#
# def broadcast_message(message: str, servers: Sequence[Server]) -> None:
#     pass
#
# broadcast_message("bla", [(("b", 2),{"k":"v"})])

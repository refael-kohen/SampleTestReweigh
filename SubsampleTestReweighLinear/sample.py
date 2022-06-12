from math import sqrt
from statistics import NormalDist
from typing import Tuple

from scipy.stats import norm

from SubsampleTestReweighLinear.utils import *


class Sample:
    def __init__(self, mean: float, std: float, mean_k: float, std_k: float, dim: int, k_coord=None) -> None:
        """
        :param mean:    expectation
        :param std:     standard deviation
        :param mean_k:  expectation of the k coordinates
        :param std_k:   standard deviation of the k coordinates
        :param dim:     dimension
        :param k_coord: list of coordinates with other parameters
        """
        if k_coord is None:
            k_coord = []
        if k_coord is None:
            k_coord = []
        self.mean = mean
        self.std = std
        self.mean_k = mean_k
        self.std_k = std_k
        self.dim = dim
        self.k_coord = k_coord

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
        return np.random.normal(loc=self.mean_arr, scale=self.std_arr, size=(int(size), self.dim))

    @staticmethod
    def get_label(sample: np.ndarray, hypothesis: np.ndarray, intercept: float) -> np.ndarray:
        """
        :param sample:     2d array
        :param hypothesis: 1d array - hypothesis without the intercept
        :param intercept:  the intercept of the hypothesis

        :return: 1d array of 1 and 0.
        """
        return np.where((sample @ hypothesis) + intercept > 0, 1, 0)

    @staticmethod
    def get_true_hypothesis(alpha: float, mean_k_t: float, std_k_t: float, dim: int, noise: str = False,
                            k_coord=None) -> np.ndarray:
        """
        :param alpha:     fraction of the samples that classified 1 in the target distribution.
        :param mean_k_t:  expectation of the target distribution in the k coordinates
        :param std_k_t:   standard deviation of the target distribution in the k coordinates
        :param dim:       dimension of the examples
        :param k_coord:   list of coordinates in T distribution with other parameters
        :param penalty:   penalty method: L2, L1, L1_NOISY. In case of L1_NOISY add normal noise to each coordinate in
                          order to became the problem harder

        :return: True hypothesis h of dimension dim+1, with 0 values in all dim coordinates except the k_coord with
                 value of 1, and in coordinate d+1 the -Z threshold value such that 1-cdf(Z) is alpha.
        """
        k = len(k_coord)
        true_hypothesis = np.zeros(dim + 1)
        true_hypothesis[k_coord + [dim]] = np.random.choice(a=[1, -1], size=k + 1)
        # cannot work with cupy
        # z_alpha = norm.ppf(q=alpha, loc=mean_k_t, scale=sqrt(k) * std_k_t)
        z_alpha = NormalDist(mu=mean_k_t, sigma=sqrt(k) * std_k_t).inv_cdf(alpha)
        true_hypothesis[dim] *= z_alpha
        if noise:
            true_hypothesis += np.random.normal(size=dim + 1)
        return np.asarray(true_hypothesis)  # in case of cupy - convert to cupy

    @staticmethod
    def get_ppf(sample_s: np.ndarray, true_hypothesis: np.ndarray, mean_k_t: float, std_k_t: float, mean_k_s: float,
                std_k_s: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the pdf of the samples from S according T distribution. The T distribution is on the scalar value of
        sample_s @ true_hypothesis, i.e. the inner product between the true hypothesis (without the b threshold
        and the samples.

        :param sample_s:            samples from S
        :param true_hypothesis :    true hypothesis on T
        :param mean_k_t:            mean of the k coordinates in T
        :param std_k_t:             std of the k coordinates in T
        :param mean_k_s:            mean of the k coordinates in S
        :param std_k_s:             std of the k coordinates in S
        :param k:                   the k differ coordinates
        :return:                    pdf of the sample of S according to S and T distribution.
        """
        # cannot work with cupy
        # pdf_sample_s_in_T = norm.pdf(x=sample_s.get() @ true_hypothesis, loc=mean_k_t, scale=sqrt(k) * std_k_t)
        pdf_sample_s = norm.pdf(x=get_np(sample_s) @ get_np(true_hypothesis[:-1]), loc=mean_k_s,
                                scale=sqrt(k) * std_k_s)
        pdf_sample_s_in_T = norm.pdf(x=get_np(sample_s) @ get_np(true_hypothesis[:-1]), loc=mean_k_t,
                                     scale=sqrt(k) * std_k_t)
        return np.asarray(pdf_sample_s), np.asarray(pdf_sample_s_in_T)

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

from typing import Callable, Tuple

import numpy

from SubsampleTestReweighLinear.sample import Sample
from SubsampleTestReweighLinear.utils import *  # Get numpy of settings as np


class SQOracle:
    def __init__(self, true_hypo: np.ndarray, sample_size: int, mean: float, std: float, mean_k: float,
                 std_k: float, v) -> None:
        """
        :param true_hypo:     true hypothesis on T distribution for the calculation of the loss
        :param sample_size:   sample size from T (or S)
        :param mean:          expectation
        :param std:           standard deviation
        :param mean_k:        expectation of the k coordinates
        :param std_k:         standard deviation of the k coordinates
        :param private:       True if private run, False non-private run
        :param rr_laplace_b:  private loss of the local mechanism.
                              Set epsilon=1 If you do not need privacy of the samples.
        """
        self.true_hypo = true_hypo
        self.mean_k = mean_k
        self.std_k = std_k
        self.sample_size = sample_size
        self.dist_oracle = Sample(mean=mean, std=std, mean_k=mean_k, std_k=std_k, dim=v.SampleV.dim,
                                  k_coord=v.SampleV.k_coord)

    def sq_query(self, loss_func: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
                 sklearn_model: int, v) -> float:
        """
        :param loss_func: Loss class that contains predictor and loss function
        :return:          estimation of the loss expectation on the target distribution
        """
        sample = self.dist_oracle.get_sample(self.sample_size)
        true_labels = self.dist_oracle.get_label(sample=sample, hypothesis=self.true_hypo[:-1],
                                                 intercept=self.true_hypo[-1])
        if sklearn_model:
            try:
                loss, loss_per_sample = loss_func(sample.get(), true_labels.get(), numpy)
            except AttributeError:
                loss, loss_per_sample = loss_func(sample, true_labels, numpy)
        else:
            loss, loss_per_sample = loss_func(sample, true_labels, np)
        if v.PrivateV.private:  # SQ with privacy
            numpy_or_cupy = numpy if sklearn_model else np
            loss_per_sample += numpy_or_cupy.random.laplace(loc=0, scale=v.PrivateV.rr_laplace_b, size=self.sample_size)
            return np.mean(loss_per_sample)
        else:
            return loss

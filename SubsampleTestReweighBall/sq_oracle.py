from typing import Callable, Tuple

from SubsampleTestReweighBall.sample import Sample
from SubsampleTestReweighBall.utils import *  # Get numpy of settings as np


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
                              Set epsilon_t=1 If you do not need privacy of the samples.
        """
        self.true_hypo = true_hypo
        self.mean_k = mean_k
        self.std_k = std_k
        self.sample_size = sample_size
        self.dist_oracle = Sample(mean=mean, std=std, mean_k=mean_k, std_k=std_k, dim=v.SampleV.dim,
                                  k_coord=v.SampleV.k_coord, limit_norm2sq=False)

    def sq_query(self, loss_func: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]], v) -> float:
        """
        :param loss_func: Loss class that contains predictor and loss function
        :return:          estimation of the loss expectation on the target distribution
        """
        sample = self.dist_oracle.get_sample(self.sample_size)
        true_labels = self.dist_oracle.get_label(sample=sample, hypothesis=self.true_hypo[:-1],
                                                 intercept=self.true_hypo[-1])
        loss_no_noise, one_loss_no_noise, loss_per_sample = loss_func(sample, true_labels)
        if v.PrivateV.private:  # SQ with privacy
            loss_per_sample += np.random.laplace(loc=0, scale=v.PrivateV.rr_laplace_b, size=self.sample_size)
            return np.mean(loss_per_sample), one_loss_no_noise, loss_no_noise
        else:
            return loss_no_noise, one_loss_no_noise, loss_no_noise

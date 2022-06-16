from typing import Callable, Type, Tuple

# noinspection PyUnresolvedReferences
import numpy as np

from SubsampleTestReweighBall.loggers import Logger
from SubsampleTestReweighBall.sq_oracle import SQOracle
from SubsampleTestReweighBall.utils import *


class LossFunc:
    def __init__(self, lrn_alg_pred: Callable[[np.ndarray], np.ndarray]):
        self.lrn_alg_pred = lrn_alg_pred

    def loss(self, sample: np.ndarray, true_labels: np.ndarray, distribution: np.ndarray = None) -> Tuple[
        float, np.ndarray]:
        """
        :param sample:          2d array with samples
        :param true_labels:     1d array with the labels
        :param numpy:           send numpy package (or cupy)
        :param distribution:    distribution of the examples (for the loss of S only - loss of T is calculated as average)
        :return:                tuple (float - mean of the loss, 1d array - with [0, 1]-loss per sample)
        """
        pred = self.lrn_alg_pred(sample)
        # pred_manually = np.where(sample @ self.hypothesis > 0, 1, 0)
        # if (pred_manually == pred.get()).all():
        #     print('yyyyyyyyyyyyeeessssssss is corrrrecccctttt !!!!')
        loss_per_sample = np.abs(pred - true_labels).astype(np.float64)
        # if distribution is None:
        #     print('losssssssssssssssssss', numpy.mean(loss_per_sample), numpy.sum(loss_per_sample), loss_per_sample @ get_np(distribution))
        #     print('losssssssssssssssssss', numpy.mean(loss_per_sample), numpy.sum(loss_per_sample))
        one_losses = np.logical_and(loss_per_sample, true_labels).astype(np.float64)
        if distribution is not None:
            return loss_per_sample @ distribution, one_losses @ distribution, loss_per_sample
        return np.mean(loss_per_sample), np.mean(one_losses), loss_per_sample


class MWTransferLearning:
    def __init__(self, model, sample: np.ndarray,
                 labels: np.ndarray, pdf_sample_t_over_s: np.ndarray,
                 sq_oracle: Type[SQOracle], log_str: str, logger: Type[Logger]) -> None:
        """
        :param model:               sklearn model or like this structure
        :param sample:              2d array. Each sample in one row
        :param labels:              2d array with the labels of sample
        :param pdf_sample_t_over_s: the pdf of samples S over the pdf of samples T
        :param sq_oracle:           instance of SQOracle class
        :param log_str:             string with parameters to print in each iteration
        :param logger:              logger

        :returns   Succeed:       (bool)         the algorithm succeed or failed (more than max_iter)
                   iter_num:      (int)          iterations number of the MW algorithm until the stop condition
                   loss_S:        (float)        mean loss on the whole training set from S distribution (test set)
                   loss_S_sub:    (float)        mean loss on the subsample from S distribution (train set)
                   loss_T:        (float)        approximation (until tau) of the mean loss on T distribution
                   kl_dist:       (np.ndarray)   KL distance between u_t and T in each step of the algorithm
                   tv_dist:       (np.ndarray)   TV distance between u_t and T in each step of the algorithm
                   chisq_dist:    (np.ndarray)   Chi-Squared distance between u_t and T in each step of the algorithm
        """
        self.model = model
        self.sample = sample
        self.labels = labels
        self.pdf_sample_t_over_s = pdf_sample_t_over_s

        self.sq_oracle = sq_oracle
        self.log_str = log_str
        self.logger = logger
        self.sample_size = self.sample.shape[0]
        self.W = np.zeros(self.pdf_sample_t_over_s.shape, dtype=np.float64)
        # self.W = np.nan_to_num(self.pdf_sample_s / self.pdf_sample_s_in_t) / np.sum(
        #     np.nan_to_num(self.pdf_sample_s / self.pdf_sample_s_in_t))
        self.W = self.pdf_sample_t_over_s / np.sum(self.pdf_sample_t_over_s)
        # self.W = np.nan_to_num(self.W)
        self.zero_loss_times = 0

    def project_weights_on_k_dense(self, weights, v):
        while True:
            sum_w = np.sum(weights)
            max_w_under_1 = np.max(np.where(weights >= 1, 0, weights))
            if (1 / max_w_under_1) * sum_w > v.PrivateV.kappa * self.sample_size:
                weights *= v.PrivateV.kappa * self.sample_size / sum_w
                return
            weights *= (1 / max_w_under_1)

    def get_chisq_dist(self, normalized_weights):
        return np.sum((self.W ** 2) / normalized_weights) - 1

    def get_kl_dist(self, normalized_weights):
        # in case of normalized_weights is zero (log is -inf) we
        # return np.sum(self.W * np.nan_to_num(np.log2(self.W / normalized_weights)))
        return np.sum(self.W * np.log2(self.W / normalized_weights))

    def get_tv_dist(self, normalized_weights):
        return 0.5 * np.sum(np.abs(self.W - normalized_weights))

    def run(self, v=None) -> Tuple[bool, int,
                                   np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: hypothesis - 1d array
        """
        self.logger.info('\t\tStart MW algorithm')
        chisq_dist = np.empty(v.MWalgV.mw_max_iter)
        kl_dist = np.empty(v.MWalgV.mw_max_iter)
        tv_dist = np.empty(v.MWalgV.mw_max_iter)
        loss_S = np.empty(v.MWalgV.mw_max_iter)
        loss_S_sub = np.empty(v.MWalgV.mw_max_iter)
        loss_T = np.empty(v.MWalgV.mw_max_iter)
        chisq_dist[:] = np.nan
        kl_dist[:] = np.nan
        tv_dist[:] = np.nan
        loss_S[:] = np.nan
        loss_S_sub[:] = np.nan
        loss_T[:] = np.nan

        T_loss_interval_avg_best = 1
        T_loss_interval_avg = 0
        weights = np.ones(self.sample_size)
        if v.PrivateV.private:  # private algorithm
            weights *= v.PrivateV.kappa
        for iter_num in range(v.MWalgV.mw_max_iter):
            self.logger.gwinfo('\t\t\tIteration number {} of MW algorithm with {}'.format(iter_num + 1, self.log_str)) \
                if iter_num % v.MWalgV.mw_print_freq == 0 else None

            if v.PrivateV.private:
                self.project_weights_on_k_dense(weights, v)
            u_t = weights / np.sum(weights)
            self.logger.info('\t\t\t\tFit model on subsample') if iter_num % v.MWalgV.mw_print_freq == 0 else None

            if v.ModelV.model_name == Const.CUSTOM_SVM:

                # #for test:
                # if v.ModelV.coef_init and iter_num == 0:
                #     self.model.fit(self.sample, self.labels, sample_dist=u_t,
                #                    coef_init=[0.16035192, 0.11679769, 0.43103747, 0.30890705, 0.47477367, 0.64860275,
                #                               0.74461949, 0.23279385],
                #                    intercept_init=0.10838071)
                #
                    # [0.16035192 0.11679769 0.43103747 0.30890705 0.47477367 0.64860275 0.74461949 0.23279385]
                    #intercept [0.10838071]

                if v.ModelV.coef_init and iter_num > 0:
                    # if v.ModelV.coef_init_cont_eta:
                    #     self.model.eta_start_iter = v.SampCompV.max_iter * (iter_num) + 1
                    # sgd algorithm returns minus of its intercept
                    self.model.fit(self.sample, self.labels, sample_dist=u_t, coef_init=self.model.coef_,
                                   intercept_init=-self.model.intercept_)
                else:
                    self.model.fit(self.sample, self.labels, sample_dist=u_t)
            else:
                subsample_idx = np.random.choice(a=self.sample_size, size=v.SampCompV.subsample_size, p=u_t,
                                                 replace=True)
                subsample = self.sample[subsample_idx]
                subsample_labels = self.labels[subsample_idx]
                self.model.fit(subsample, subsample_labels)
            ##################
            # w = Model.model.best_estimator_.steps[0][1].coef_
            # intercept = Model.model.best_estimator_.steps[0][1].intercept_
            # hypothesis = np.append(w, intercept)
            # best_params_S = Model.model.best_params_
            # best_score_S = Model.model.best_score_
            ##################
            for log in self.model.logs:
                if log:
                    self.logger.info('\t\t\t\t\t{}'.format(log))
            self.logger.info('\t\t\t\t\t-- Model coefficients {}, intercept {}'.format(self.model.coef_,
                                                                                       -self.model.intercept_)) if iter_num % v.MWalgV.mw_print_freq == 0 else None
            self.logger.winfo('\t\t\t\t\t-- Model coefficients {}, intercept {}'.format(self.model.coef_,
                                                                                        -self.model.intercept_)) if iter_num % v.MWalgV.mw_print_freq == 0 else None

            self.logger.info('\t\t\t\tCalculate loss on S') if iter_num % v.MWalgV.mw_print_freq == 0 else None
            loss_func = LossFunc(lrn_alg_pred=self.model.predict)
            loss_S[iter_num], one_loss_s, loss_per_sample_S = loss_func.loss(self.sample, self.labels, u_t)
            loss_per_sample_S = loss_per_sample_S
            if v.ModelV.model_name != Const.CUSTOM_SVM:
                loss_S_sub[iter_num], _, loss_per_sample_Sub = loss_func.loss(subsample, subsample_labels)
            if v.ModelV.model_name == Const.CUSTOM_SVM:
                loss_S_sub[iter_num] = -1
            self.logger.info('\t\t\t\tSQ - query') if iter_num % v.MWalgV.mw_print_freq == 0 else None
            # in non-private loss_T[iter_num] and loss_t_no_noise are the same
            loss_T[iter_num], one_loss_t, loss_t_no_noise = self.sq_oracle.sq_query(loss_func.loss, v)
            chisq_dist[iter_num] = self.get_chisq_dist(u_t)
            kl_dist[iter_num] = self.get_kl_dist(u_t)
            tv_dist[iter_num] = self.get_tv_dist(u_t)
            self.logger.info('\t\t\t\tLoss S: {}, Loss subsampling S: {}, '
                             'Loss T: {}, Dist chisq: {}, Dist kl: {}, Dist tv: {}'.format(loss_S[iter_num],
                                                                                           loss_S_sub[iter_num],
                                                                                           loss_T[iter_num],
                                                                                           chisq_dist[iter_num],
                                                                                           kl_dist[iter_num], tv_dist[
                                                                                               iter_num])) if iter_num % v.MWalgV.mw_print_freq == 0 else None
            self.logger.info(
                '\t\t\t\tSample S has fraction of label 0: {}, 0-loss S: {}, 1-loss-S: {}, 0-loss T: {}, 1-loss-T: {}  '.format(
                    1 - float(self.labels @ u_t), loss_S[iter_num] - one_loss_s, one_loss_s,
                    loss_t_no_noise - one_loss_t, one_loss_t))

            T_loss_interval_avg += loss_T[iter_num]
            if iter_num > 1 and iter_num % v.MWalgV.n_iter_no_change == 0:
                T_loss_interval_avg /= v.MWalgV.n_iter_no_change
                self.logger.info(
                    '\t\t\t\tAverage T loss in last interval: {}, best average T loss in intervals of size {} is: {}'.format(
                        T_loss_interval_avg, v.MWalgV.n_iter_no_change, T_loss_interval_avg_best))
                if (T_loss_interval_avg_best - T_loss_interval_avg < v.MWalgV.tol_early_stopping) \
                        and v.MWalgV.early_stopping:
                    self.logger.info('\t\tEarly stopping')
                    return True, iter_num, loss_S, loss_S_sub, loss_T, kl_dist, tv_dist, chisq_dist
                T_loss_interval_avg_best = T_loss_interval_avg
                T_loss_interval_avg = 0

            # if the loss on S is 0 the weights will be not updated in the next iterations
            if loss_S[iter_num] == 0:
                self.zero_loss_times += 1
            else:
                self.zero_loss_times = 0
            if self.zero_loss_times > v.MWalgV.zero_loss_times_max:
                return False, iter_num, loss_S, loss_S_sub, loss_T, kl_dist, tv_dist, chisq_dist
            if loss_T[iter_num] > v.MWalgV.stop_condition:
                weights *= np.exp(-v.MWalgV.eta * (1 - loss_per_sample_S))
                # weights = np.nan_to_num(weights)
                # self.logger.wdebug('\t\t\t\tWeights are: {}'.format(weights)) if iter_num % 20 == 0 else None
            else:
                return True, iter_num, loss_S, loss_S_sub, loss_T, kl_dist, tv_dist, chisq_dist

        # The algorithm failed
        if v.MWalgV.simulation_mode:
            return False, iter_num, loss_S, loss_S_sub, loss_T, kl_dist, tv_dist, chisq_dist
        else:
            raise Exception("Too many iterations")

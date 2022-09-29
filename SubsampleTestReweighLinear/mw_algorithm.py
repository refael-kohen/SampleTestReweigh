from typing import Callable, Type, Tuple

import numpy
# noinspection PyUnresolvedReferences
from sklearn.model_selection import GridSearchCV

from SubsampleTestReweighLinear.loggers import Logger
from SubsampleTestReweighLinear.sq_oracle import SQOracle
from SubsampleTestReweighLinear.utils import *  # import np from utils


class LossFunc:
    def __init__(self, lrn_alg_pred: Callable[[np.ndarray], np.ndarray]):
        self.lrn_alg_pred = lrn_alg_pred
        # self.hypothesis = np.asarray(hypothesis)

    def loss(self, sample: np.ndarray, true_labels: np.ndarray, numpy, distribution: np.ndarray = None) -> Tuple[
        float, np.ndarray]:
        """
        :param sample:          2d array with samples
        :param true_labels:     1d array with the labels
        :param numpy:           send numpy package (or cupy)
        :param distribution:    distribution of the examples (for the loss of S)
        :return:                tuple (float - mean of the loss, 1d array - with [0, 1]-loss per sample)
        """
        pred = self.lrn_alg_pred(sample)
        # pred_manually = np.where(sample @ self.hypothesis > 0, 1, 0)
        # if (pred_manually == pred.get()).all():
        #     print('yyyyyyyyyyyyeeessssssss is corrrrecccctttt !!!!')
        loss_per_sample = numpy.abs(pred - true_labels).astype(np.float64)
        if distribution is not None:
            return float(loss_per_sample @ get_np(distribution)), loss_per_sample
        return float(numpy.mean(loss_per_sample)), loss_per_sample


class MWTransferLearning:
    def __init__(self, model, sample: np.ndarray,
                 labels: np.ndarray, pdf_sample_s_in_t: np.ndarray, pdf_sample_s: np.ndarray,
                 sq_oracle: Type[SQOracle], log_str: str, logger: Type[Logger]) -> None:
        """
        :param model:             sklearn model or like this structure
        :param sample:            2d array. Each sample in one row
        :param labels:            2d array with the labels of sample
        :param pdf_sample_s_in_t: the pdf of samples from S according T distribution.
        :param pdf_sample_s:      the pdf of samples S
        :param sq_oracle:         instance of SQOracle class
        :param log_str:           string with parameters to print in each iteration
        :param logger:            logger

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
        self.pdf_sample_s_in_t = pdf_sample_s_in_t
        self.pdf_sample_s = pdf_sample_s
        self.sq_oracle = sq_oracle
        self.log_str = log_str
        self.logger = logger
        self.sample_size = self.sample.shape[0]
        self.W = (self.pdf_sample_s_in_t / self.pdf_sample_s)
        self.W /= np.sum(self.W)
        self.zero_loss_times = 0

    def project_weights_on_k_dense(self, weights):
        while True:
            sum_w = np.sum(weights)
            max_w_under_1 = np.max(np.where(weights >= 1, 0, weights))
            if (1 / max_w_under_1) * sum_w > v.PrivateV.private * self.sample_size:
                weights *= v.PrivateV.private * self.sample_size / sum_w
                return
            weights *= (1 / max_w_under_1)

    def get_chisq_dist(self, weights):
        return np.sum((self.W ** 2) / weights) - 1

    def get_kl_dist(self, weights):
        # in case of self.W is zero (log is -inf) we
        return np.sum(self.W * np.nan_to_num(np.log2(self.W / weights)))

    def get_tv_dist(self, weights):
        return 0.5 * np.sum(np.abs(self.W - weights))

    def run(self, print_freq=1, v=None) -> Tuple[bool, int,
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
            weights *= v.PrivateV.private
        for iter_num in range(v.MWalgV.mw_max_iter):
            self.logger.gwinfo('\t\t\tIteration number {} of MW algorithm with {}'.format(iter_num + 1, self.log_str)) \
                if iter_num % print_freq == 0 else None

            if v.PrivateV.private:
                self.project_weights_on_k_dense(weights)
            u_t = weights / np.sum(weights)
            subsample_idx = numpy.random.choice(a=self.sample_size, size=v.SampCompV.subsample_size, p=get_np(u_t),
                                                replace=True)
            subsample = self.sample[subsample_idx]
            subsample_labels = self.labels[subsample_idx]
            self.logger.info('\t\t\t\tFit model on subsample') if iter_num % print_freq == 0 else None
            # TODO: SVM run only on 100 coordinates, and find hypothesis of size d (in addition to the intercept).
            # scikit learn cannot work with cupy
            if v.RunV.sklearn_model:
                self.model.fit(get_np(subsample), get_np(subsample_labels))
            else:
                self.model.fit(subsample, subsample_labels)
            ##################
            # w = Model.model.best_estimator_.steps[0][1].coef_
            # intercept = Model.model.best_estimator_.steps[0][1].intercept_
            # hypothesis = np.append(w, intercept)
            # best_params_S = Model.model.best_params_
            # best_score_S = Model.model.best_score_
            ##################
            self.logger.info('\t\t\t\tCalculate loss on S') if iter_num % print_freq == 0 else None
            loss_func = LossFunc(lrn_alg_pred=self.model.predict)
            # scikit learn cannot work with cupy
            if v.RunV.sklearn_model:
                loss_S[iter_num], loss_per_sample_S = loss_func.loss(get_np(self.sample), get_np(self.labels),
                                                                     numpy, u_t)
                loss_S_sub[iter_num], loss_per_sample_Sub = loss_func.loss(get_np(subsample),
                                                                           get_np(subsample_labels), numpy)
                loss_per_sample_S = np.asarray(loss_per_sample_S)  # converts to cupy if np is cupy
                loss_per_sample_Sub = np.asarray(loss_per_sample_Sub)  # converts to cupy if np is cupy
            else:
                loss_S[iter_num], loss_per_sample_S = loss_func.loss(self.sample, self.labels, np, u_t)
                loss_S_sub[iter_num], loss_per_sample_Sub = loss_func.loss(subsample, subsample_labels, np)
            self.logger.info('\t\t\t\tSQ - query') if iter_num % print_freq == 0 else None
            loss_T[iter_num] = self.sq_oracle.sq_query(loss_func.loss, v.RunV.sklearn_model, v)
            chisq_dist[iter_num] = self.get_chisq_dist(u_t)
            kl_dist[iter_num] = self.get_kl_dist(u_t)
            tv_dist[iter_num] = self.get_tv_dist(u_t)
            self.logger.info('\t\t\t\tLoss S: {}, Loss subsampling S: {}, '
                             'Loss T: {}, Dist chisq: {}, Dist kl: {}, Dist tv: {}'.format(loss_S[iter_num],
                                                                                           loss_S_sub[iter_num],
                                                                                           loss_T[iter_num],
                                                                                           chisq_dist[iter_num],
                                                                                           kl_dist[iter_num], tv_dist[
                                                                                               iter_num])) if iter_num % print_freq == 0 else None
            T_loss_interval_avg += loss_T[iter_num]
            if iter_num > 1 and iter_num % v.MWalgV.n_iter_no_change == 0:
                T_loss_interval_avg /= v.MWalgV.n_iter_no_change
                self.logger.info(
                    '\t\t\t\tAverage T loss in last interval: {}, best average T loss in intervals of size {} is: {}'.format(
                        T_loss_interval_avg, v.MWalgV.n_iter_no_change, T_loss_interval_avg_best))
                if (
                        T_loss_interval_avg_best - T_loss_interval_avg < v.MWalgV.tol_early_stopping) and v.MWalgV.early_stopping:
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
                weights *= np.exp(-v.PrecV.eta * (1 - loss_per_sample_S))
                weights = np.nan_to_num(weights)
                # self.logger.wdebug('\t\t\t\tWeights are: {}'.format(weights)) if iter_num % 20 == 0 else None
                self.logger.wdebug('\t\t\t\tModel coefficients {}, intercept {}'.format(self.model.coef_,
                                                                                        self.model.intercept_)) if iter_num % 20 == 0 else None
            else:
                return True, iter_num, loss_S, loss_S_sub, loss_T, kl_dist, tv_dist, chisq_dist

        # The algorithm failed
        if v.MWalgV.simulation_mode:
            return False, iter_num, loss_S, loss_S_sub, loss_T, kl_dist, tv_dist, chisq_dist
        else:
            raise Exception("Too many iterations")

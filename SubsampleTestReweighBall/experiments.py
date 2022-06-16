# noinspection PyUnresolvedReferences
import multiprocessing
# noinspection PyUnresolvedReferences
from math import ceil, log2, log, exp, sqrt

# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
from scipy.stats import chi2
from sklearn.linear_model import SGDClassifier
# noinspection PyUnresolvedReferences
from sklearn.svm import LinearSVC

from SubsampleTestReweighBall.LP_model import LP
from SubsampleTestReweighBall.data_structures import DataStruct
from SubsampleTestReweighBall.loggers import Logger
from SubsampleTestReweighBall.mw_algorithm import MWTransferLearning
from SubsampleTestReweighBall.sample import Sample
from SubsampleTestReweighBall.sgd_classifier_custom import SGDClassifierCustom
# noinspection PyUnresolvedReferences
from SubsampleTestReweighBall.sq_oracle import SQOracle
from SubsampleTestReweighBall.utils import *  # import np from utils


def run_repetition(v, model, log_str, ds_succeeded, ds_failed,
                   return_dict, rep, logger=None, rep_counter=None, queue=None, configurer=None):
    # Initialize multiprocessing (on linux) each process inherits the random state from the parent.
    np.random.seed()
    if v.RunV.multiproc_num:
        configurer(queue)
        logger = Logger(rep_counter, v=v)

    logger.gwinfo('\tStart repetition {} with {}'.format(rep + 1, log_str))
    # logger.info('\t\tPrepare the source distribution')
    logger.info('\t\tSample from the source distribution {} examples'.format(v.SampCompV.sample_size_s))
    source_oracle = Sample(mean=v.SampleV.mean_s, std=v.SampleV.std_s, mean_k=v.SampleV.mean_k_s, std_k=v.SampleV.std_k_s,
                      k_coord=v.SampleV.k_coord, dim=v.SampleV.dim, limit_norm2sq=v.ModelV.limit_norm2sq,
                      norm2sq=v.ModelV.norm2sq)
    sample_s = source_oracle.get_sample(v.SampCompV.sample_size_s)
    true_hypo = source_oracle.get_true_hypothesis(v)
    logger.info('\t\tThe real hypothesis is: {}'.format(true_hypo))

    logger.info('\t\tCalculate pdf of the source distribution')
    pdf_sample_t_over_s = Sample.get_pdf(sample_s=sample_s, true_hypothesis=true_hypo,
                                                     mean_k_t=v.SampleV.mean_k_t, std_k_t=v.SampleV.std_k_t,
                                                     mean_k_s=v.SampleV.mean_k_s, std_k_s=v.SampleV.std_k_s,
                                                     k=v.SampleV.k, k_coord=v.SampleV.k_coord)
    logger.info('\t\tGet labels of the source distribution')
    labels_s = Sample.get_label(sample=sample_s, hypothesis=true_hypo[:-1], intercept=true_hypo[-1])
    one_labeled_s = np.sum(labels_s)
    zero_labeled_s = v.SampCompV.sample_size_s - np.sum(labels_s)
    logger.info('\t\tSample S has {} examples with label 0 ({}), and {} examples with label 1 ({})'.format(
        zero_labeled_s, zero_labeled_s / v.SampCompV.sample_size_s, one_labeled_s,
                        one_labeled_s / v.SampCompV.sample_size_s))

    # In non-privacy mode set 1 in epsilon_T
    sq_oracle = SQOracle(true_hypo=true_hypo, sample_size=v.SampCompV.sample_size_t, mean=v.SampleV.mean_t,
                         std=v.SampleV.std_t, mean_k=v.SampleV.mean_k_t, std_k=v.SampleV.std_k_t, v=v)

    dist_oracle = Sample(mean=v.SampleV.mean_t,
                         std=v.SampleV.std_t, mean_k=v.SampleV.mean_k_t, std_k=v.SampleV.std_k_t, dim=v.SampleV.dim,
                         k_coord=v.SampleV.k_coord, limit_norm2sq=v.ModelV.limit_norm2sq, norm2sq=v.ModelV.norm2sq)
    sample_t = dist_oracle.get_sample(v.SampCompV.sample_size_t)
    labels_t = dist_oracle.get_label(sample=sample_t, hypothesis=true_hypo[:-1], intercept=true_hypo[-1])

    one_labeled_t = np.sum(labels_t)
    zero_labeled_t = v.SampCompV.sample_size_t - np.sum(labels_t)
    logger.info('\t\tSample T has {} examples with label 0 ({}), and {} examples with label 1 ({})'.format(
        zero_labeled_t, zero_labeled_t / v.SampCompV.sample_size_t, one_labeled_t,
                        one_labeled_t / v.SampCompV.sample_size_t))

    mw_algo = MWTransferLearning(model=model, sample=sample_s, labels=labels_s, pdf_sample_t_over_s=pdf_sample_t_over_s,
                                 sq_oracle=sq_oracle, log_str='rep: {}, '.format(rep + 1) + log_str, logger=logger)
    succeeded, iter_num_idx, loss_s, loss_s_sub, loss_t, kl_dist, tv_dist, chisq_dist = mw_algo.run(v=v)
    logger.info('\tFinish repetition {} of MW algorithm after {} iterations in status {}: Loss S: {}, '
                'Loss subsampling S: {}, Loss T: {}, chisq_dist: {}, kl_dist: {}, '
                'tv_dist: {} with {}'.format(rep + 1, iter_num_idx + 1, 'succeeded' if succeeded else 'failed',
                                             loss_s[iter_num_idx], loss_s_sub[iter_num_idx],
                                             loss_t[iter_num_idx], chisq_dist[iter_num_idx],
                                             kl_dist[iter_num_idx], tv_dist[iter_num_idx], log_str))

    # Because the program can be parallel the updating of the files need to be in each parallel repetition, in order to
    # print the result of the repetition immediately in the end of repetition and not wait to end of all repetitions.
    if succeeded:
        ds_succeeded.print_data_repetition(1, iter_num_idx + 1, chisq_dist, kl_dist, tv_dist, loss_s, loss_s_sub,
                                           loss_t, rep_counter, v)
        return_dict[rep + 1] = 1
    else:
        ds_failed.print_data_repetition(0, iter_num_idx + 1, chisq_dist, kl_dist, tv_dist, loss_s, loss_s_sub, loss_t,
                                        rep_counter, v)
        return_dict[rep + 1] = 0


def create_model(v):
    model = None
    if v.ModelV.model_name == Const.CUSTOM_SVM:
        if v.PrivateV.private:
            model = SGDClassifierCustom(early_stopping=v.ModelV.early_stopping,
                                        early_stopping_score=v.ModelV.early_stopping_score,
                                        early_stopping_n_iter=v.ModelV.early_stopping_n_iter,
                                        private=True, fit_intercept=True, alpha=1.0 / v.ModelV.reg_c,
                                        max_iter=v.SampCompV.max_iter, average=v.ModelV.average_coef,
                                        loss='circle_hinge', eta0=v.SampCompV.eta_sgd,
                                        eta_start_iter=v.ModelV.eta_start_iter, mean_noise=0.0,
                                        sigam_noise=v.SampCompV.ss_noise_sigma, batch_size=v.ModelV.batch_size,
                                        do_proj=v.ModelV.do_proj, min_proj=v.ModelV.min_proj,
                                        max_proj=v.ModelV.max_proj, random_blocks_num=v.ModelV.random_blocks_num,
                                        verbose=v.ModelV.verbose)
        else:
            # model = LinearSVC(C=1000000000000, fit_intercept=True, dual=False)
            model = SGDClassifierCustom(early_stopping=v.ModelV.early_stopping,
                                        early_stopping_score=v.ModelV.early_stopping_score,
                                        early_stopping_n_iter=v.ModelV.early_stopping_n_iter,
                                        private=False, fit_intercept=True, alpha=1.0 / v.ModelV.reg_c,
                                        max_iter=v.SampCompV.max_iter, average=v.ModelV.average_coef,
                                        loss='circle_hinge', eta0=v.SampCompV.eta_sgd,
                                        eta_start_iter=v.ModelV.eta_start_iter,
                                        mean_noise=0.0, sigam_noise=0.0, batch_size=v.ModelV.batch_size,
                                        do_proj=v.ModelV.do_proj, min_proj=v.ModelV.min_proj,
                                        max_proj=v.ModelV.max_proj, random_blocks_num=v.ModelV.random_blocks_num,
                                        verbose=v.ModelV.verbose)
            # learning_rate='constant', eta0=5
    elif v.ModelV.model_name == Const.LR:  # logistic regression
        # model = LogisticRegression(C=REGULARIZATION_C, n_jobs=1, fit_intercept=True)
        model = SGDClassifier(loss='log', fit_intercept=True, penalty=v.ModelV.penalty,
                              alpha=1 / v.ModelV.reg_c)
    elif v.ModelV.model_name == Const.LP:  # linear programming
        model = LP()
    return model


def run_experiments(v):
    logger = Logger(v=v)
    frac_zero_label_t_thres = chi2.ppf(v.SampleV.frac_zero_label_t, v.SampleV.k)
    logger.info('std_k_t: {}, frac_zero_label_t: {}, frac_zero_label_t_thres: {}'.format(v.SampleV.std_k_t,
                                                                                         v.SampleV.frac_zero_label_t,
                                                                                         frac_zero_label_t_thres))
    model = create_model(v)

    ds_succeeded = DataStruct()
    ds_failed = DataStruct()

    rep_counter = 1

    logger.info('The upper bound on sample S is: {}'.format(v.SampCompV.sample_size_s_upper_bound))
    log_str = 'MW_max_iter: {}, sample_size_t: {}, sample_size_s: {}, ' \
              'subsample_size: {}'.format(v.MWalgV.mw_max_iter, v.SampCompV.sample_size_t, v.SampCompV.sample_size_s,
                                          v.SampCompV.subsample_size)

    logger.gwinfo('Start simulation with {}'.format(log_str))

    args = (v, model, log_str, ds_succeeded, ds_failed)
    if v.RunV.multiproc_num:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()  # hold the failed/succeed repetition (filled by the process)
        # https://docs.python.org/3/howto/logging-cookbook.html
        # asyncronic parallel
        # https://opensourceoptions.com/blog/asynchronous-parallel-programming-in-python-with-multiprocessing/
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=Logger.listener_process, args=(queue, Logger.listener_configurer))
        listener.start()
        for s in range(0, v.RunV.num_rep, v.RunV.multiproc_num):
            workers = []
            for rep in range(s, s + v.RunV.multiproc_num):
                args2 = (return_dict, rep, None, rep_counter, queue, Logger.worker_configurer)

                worker = multiprocessing.Process(target=run_repetition, args=(*(args + args2),))
                workers.append(worker)
                worker.start()
                rep_counter += 1
            for w in workers:
                w.join()

        queue.put_nowait(None)
        listener.join()
    else:
        return_dict = {}
        for rep in range(v.RunV.num_rep):
            args2 = (return_dict, rep, logger)
            run_repetition(*(args + args2), )
    succeeded_rep_num = sum(return_dict.values())
    logger.info(
        '\tFinish {} succeeded repetitions (out of {}) with {}'.format(succeeded_rep_num, v.RunV.num_rep, log_str))

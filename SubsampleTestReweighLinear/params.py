# noinspection PyUnresolvedReferences
from logging import DEBUG, INFO, ERROR
# noinspection PyUnresolvedReferences
from math import ceil, log2, exp, log, sqrt
from SubsampleTestReweighLinear.utils import *

# Parameters for fast test:#
# v.SampleV.sample_size = 90000
# v.PrecV.alpha = 0.01  # 0.01
# v.RunV.num_rep = 5  # repeats of run for each combination of the parameters
# v.ModelV.model = 'LR'  # LP, LR, SVM # linear programming/ logistic regression / LinearSVC
# v.ModelV.penalty = Const.L1  # panelty method (for scikit-learn model - SGDClassifier): l2, l1
# v.ModelV.real_hypo_noise = True


# Sample
class SampleVC:
    dim = 500  # must be even (in _sgd_fast.pyx)
    mean_s = 0
    std_s = 1
    mean_k_s = 0
    std_k_s = 1
    mean_t = 0
    std_t = 1
    k_coord = list(range(40, 50))
    k = len(k_coord)
    mean_k_t = 0
    std_k_t = 0.01  # sqrt 10 #1.341  # sqrt(1.8)
    chi_s = (5 / 3) ** k


class PrecVC:
    alpha_h = 0  # the error on s of the best hypothesis
    beta = 0.05
    # sq oracle
    tau = 0.01  # error of sq_query
    alpha = 0.01  # 0.01 # error on s
    eta = alpha  # 'alpha / 8' # learning rate


class ModelVC:
    model_name = Const.SVM #Const.SVM  # LP, LR, SVM # linear programming/ logistic regression / LinearSVC
    penalty = Const.L2  # PENALTY METHOD (For scikit-learn model - SGDClassifier): L2, L1
    real_hypo_noise = False  # Add noise to real hypothesis in order to make the problem harder (useful for SVM with L1 penalty) - in fact the problem became easier !!!!
    regularization_c = 1000000000000000000000000000000  # 0.0001 #0.000000000001 # regularization parameter of SGDClassifier the default is: (alpha=0.0001, C=10000) alpha = 1/C, the lower the more precision classifier but also much more slower


class PrivateVC:
    # Private algorithm
    private = False  # True - private run, False - non-private furn
    kappa = 0.000007  # 0.1  # class MWTransferLearning. TODO: What the value need to be ?
    epsilon = 0.5  # class SQOracle. 1 in non-private algorithm. TODO: What the value need to be ?
    rr_laplace_b = 1 / (25 * epsilon)  # 2 / EPSILON # b parameter of laplace noise for randomized-response (sq-query)

    # Private subsampling parameters
    ss_noise_sigma = 4.48  # > sqrt(20) = 4.47
    ss_noise_debug_mode = True  # print from _sgd_fast.pyx file
    ss_noise_max_iter = max((ss_noise_sigma ** 2) * sqrt(SampleVC.dim) / (PrecVC.alpha ** 2),
                            log(2 / PrecVC.beta) / PrecVC.alpha ** 2)
    ss_noise_num_threads_rand = 0  # number of threads for SGD algorithm. Can slow down the program. use only when DIM is very very large.


# MW algorithm
class MWalgVC:
    """
    :param max_iter:          maximum iteration to run the algorithm
    :param zero_loss_times_max: maximum times that the loss is zero on S sample - above this value the algo' fails.
    :param stop_condition:    stop condition of the algorithm is when loss on T is lower equals to "stop_condition"
    :param early_stopping:    Whether to use early stopping to terminate training when validation score is not improving.
    :param simulation_mode:   if True, don't raise exception when iteration number exceed the maximum
    :param n_iter_no_change:  Number of iterations with no improvement to wait before stopping fitting (if early_stopping==True).
    :param tol_early_stopping:The stopping criterion. The training will stop when
                              (best_T_loss - min_T_loss_in_interaval < tol) for interval of n_iter_no_change.
    """
    stop_condition = 2 * PrecVC.alpha # + PrecVC.tau + PrecVC.alpha_h # We try to make it even worse and set lower stopping conditions
    p_dim = SampleVC.dim  # For 0-1 classification p-dim = vc-dim. For halfspace algorithm vc-dim = dim
    mw_max_iter = 10000  # ceil(64 * log2(8 * (v.SampleV.chi_s + 1)) / (v.PrecV.alpha ** 2)) # expected number of MW algorithm iterations depending on alpha

    zero_loss_times_max = 5  # After such number of iterations in which loss on S is 0 - the algorithm fails.
    early_stopping = False
    n_iter_no_change = 200
    tol_early_stopping = 0.01

    simulation_mode = True  # don't raise exception when MW exceed the max iteration (returns statistics).
    mw_print_freq = 1  # logging to stdout each such number of iterations of MW algorithm


# The real sample complexity of S
# MAX_SAMPLE_SIZE_S_MIN_ALPHA_TEMP = 'ceil((800 * (CHI_S + 1) * log2(1 / MIN_ALPHA) / MIN_ALPHA ** 2) * (P_DIM * log2(400 * P_DIM / MIN_ALPHA ** 3) + ' \
#                 'log2(8 / BETA))) '

# min(MAX_SAMPLE_SIZE, asymptotic of MAX_SAMPLE_SIZE_S_MIN_ALPHA_TEMP)
# The real sample complexity of S without constants and with the MIN_ALPHA
class SampCompVC:
    subsample_size = ceil((SampleVC.dim + log(MWalgVC.mw_max_iter / PrecVC.beta)) / (PrecVC.alpha + PrecVC.alpha_h))
    # sample_size_s = ceil((800 * (v.SampleV.chi_s + 1) * log2(1 / v.PrecV.alpha) / v.PrecV.alpha ** 2) * (
    #                            v.MWalgV.p_dim * log2(400 * v.MWalgV.p_dim / v.PrecV.alpha ** 3) + log2(8 / v.PrecV.beta)))
    sample_size_s = 50000
    sample_size_t = 50000  # ceil(2 * log(2*v.MWalgV.mw_max_iter / v.PrecV.beta) / ((v.PrecV.epsilon ** 2) * (v.PrecV.tau ** 2)))


class RunVC:
    """
        :param sklearn_model:  if True convert cupy arrays to numpy before using function of scikit-learn
    """
    multiproc_num = 0  # number of parrallel runs. 0 is not parallel, n is number of repetitions to run in parallel
    # must to divide the rep_num
    num_rep = 50  # repeats of run for each combination of the parameters
    if multiproc_num:
        save_memory = True
        no_summary = True

    log_level = DEBUG
    log_level_stdout = DEBUG
    sklearn_model = True  # convert cupy to numpy before sending to the model


class PathVC:
    date = None
    title = None
    output_root = None
    log_path = None
    plot_path = None
    output_path = None
    title_path = None

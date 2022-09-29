# noinspection PyUnresolvedReferences
from logging import DEBUG, INFO, ERROR
# noinspection PyUnresolvedReferences
from math import ceil, log2, exp, log, sqrt
import numpy as np
from SubsampleTestReweighBall.utils import *

private_params = True


# Parameters for fast test:#
# v.SampleV.sample_size = 90000
# v.PrecV.alpha = 0.01  # 0.01
# v.RunV.num_rep = 5  # repeats of run for each combination of the parameters
# v.ModelV.model = 'LR'  # LP, LR, SVM # linear programming/ logistic regression / LinearSVC
# v.ModelV.penalty = Const.L1  # panelty method (for scikit-learn model - SGDClassifier): l2, l1
# v.ModelV.real_hypo_noise = True


# Private run:
# ------------
# Fix R and T and n (according the formula that depends on R and T and epsilon_s), and change epsilon_s from 1 and above.

class PrecVC:
    alpha_h = 0.0  # the error of the best hypothesis on S
    beta = 'np.exp(-6)' # 0.05  # 10-5 or 1/1000 or e^-7
    # sq oracle
    tau = 0.01  # error of sq_query # influence on sample_size_t - the bigger tau the lower sample_size_t
    alpha = 0.13  # 0.01 # error on T


# Sample
class SampleVC:
    dim = 8   # must be even (in _sgd_fast.pyx)
    mean_s = 0.0
    std_s = 0.1
    mean_k_s = 0.0
    std_k_s = 0.1
    mean_t = 0.0
    std_t = 0.1
    k_coord = list(range(5, 7))
    frac_zero_label_t = 0.4  # fraction of examples in T that are 0-labeled (according that we consider the threshold)
    k = 'len(v.SampleV.k_coord)'
    mean_k_t = 0.0
    # for k=5, 1/sqrt(r), r=chi2.ppf(frac_zero_label_t, k)=chi2.ppf(0.08, 5)=1.4390002559287456
    # we requires the intercept will be in range [0,1]: r*sigma^2 <= 1   ==>    sigma <= 1/sqrt(r) = 1/sqrt(1.439)
    std_k_t = 'min(1/sqrt(chi2.ppf(v.SampleV.frac_zero_label_t, v.SampleV.k)), v.SampleV.std_k_s)'
    chi_s = '(v.SampleV.std_s**4/(v.SampleV.std_k_t**2 * (2 * v.SampleV.std_s**2 - v.SampleV.std_k_t**2))) ** (v.SampleV.k/2)'  # it is chisq+1 !!!


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
    stop_condition = '2 * v.PrecV.alpha'  # + PrecVC.tau' # + PrecVC.alpha_h # We try to make it even worse and set lower stopping conditions
    p_dim = 'v.SampleV.dim'  # For 0-1 classification p-dim = vc-dim. For halfspace algorithm vc-dim = dim
    if private_params:
        mw_max_iter = 1000
    else:
        mw_max_iter = 3000  # 'ceil(64 * log2(8 * (v.SampleV.chi_s + 1)) / (v.PrecV.alpha ** 2))' # expected number of MW algorithm iterations depending on alpha
    eta = 'v.PrecV.alpha/8'  # 'alpha / 8' # learning rate

    zero_loss_times_max = 20  # After such number of iterations in which loss on S is 0 - the algorithm fails.
    if private_params:
        early_stopping = False
    else:
        early_stopping = False
    n_iter_no_change = 200
    tol_early_stopping = 0.01

    simulation_mode = True  # don't raise exception when MW exceed the max iteration (returns statistics).
    mw_print_freq = 1  # logging to stdout each such number of iterations of MW algorithm


class ModelVC:
    model_name = Const.CUSTOM_SVM  # Const.CUSTOM_SVM # CUSTOM_SVM, LP, LR, SVM # CUSTOM_SVM is our implementation of SVM, linear programming/ logistic regression / LinearSVC
    penalty = Const.L2  # PENALTY METHOD (For scikit-learn model - SGDClassifier): L2, L1 - Not in use for Const.CUSTOM_SVM
    real_hypo_noise = False  # Not in use - Add noise to real hypothesis in order to make the problem harder (useful for SVM with L1 penalty) - in fact the problem became easier !!!!
    reg_c = 1.0  # 0.0001 #0.000000000001 # regularization parameter of SGDClassifier the default is: (alpha=0.0001, C=10000) alpha = 1/C, the lower the more precision classifier but also much more slower
    # diameter is the diameter of the 0-1 hypotheses class (including the intercept)
    diameter = 'sqrt(v.SampleV.dim +1)'
    intercept_raw = 'chi2.ppf(v.SampleV.frac_zero_label_t, v.SampleV.k) * v.SampleV.std_k_t ** 2'
    # Old methods:
    # 1. bound of norm1 of chiseq distiribution - Notice !!! it is unused circular formula that depends on sample_size_s->max_iter->lipschitz->B->sample_size_s
    # B = 'v.SampleV.dim + 2*sqrt(v.SampleV.dim*log(sample_size_s/v.PrecV.beta)) + 2*log(v.SampCompV.sample_size_s/v.PrecV.beta)'
    # 2. bound on bound of norm1 of chiseq distribution under assumption of small sample_size_s (validation made in run_str_simulation.py)
    # B = '2 * v.SampleV.dim'
    # New method (need to change the functions: empirical_bound_on_examples, get_sample)
    # 3. Empirical bound: Upper bound (w.p. of 1-v.ModelV.norm2sq_percentage) on the norm2 squared of the examples (the squared examples)
    limit_norm2sq = True  # limit the norm2sq of the examples from S and T
    norm2sq = 'empirical'
    norm2sq_percentage = 0.1
    if private_params:
        # In non-private the values of norm2sq are small (we use small std_k_t), so we prefer use +norm2sq
        lipschitz = 'v.ModelV.reg_c*sqrt(v.ModelV.norm2sq + v.ModelV.norm2sq)'
        # Used only for checking that it is not above 1 (in validate_input_params() fucntion)
        intercept = 'v.ModelV.intercept_raw / np.sqrt(v.ModelV.norm2sq)'
    else:
        # In non-private the values of norm2sq are above 1, so we prefer use +1
        lipschitz = 'v.ModelV.reg_c*sqrt(v.ModelV.norm2sq + 1)'
        intercept = 'v.ModelV.intercept_raw'
    batch_size = 1
    do_proj = 1  # 1 or 0, do or not projection on set
    min_proj = 0.0 # min value of the set
    max_proj = 1.0  # max value of the set
    if private_params:
        # stop the SGD algorithm when the error on SampCompVC.sample_size_s is under early_stopping_score (chek it each early_stopping_n_iter iterations)
        early_stopping = False  # must be false - we cannot check the loss on S
        # 'average_coef: When set to 1 or True, computes the averaged SGD weights across all updates and stores the result in the coef_ attribute. If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.'
        average_coef = True # because early stopping is False you need return average of the coef, because the last one don't work good on S
        eta_start_iter = 1 # Start from later eta (simulate that several iteration already run - it is don't harm the privacy and good if we take the average of the hypotheses and want to ignore the first iterations).
        coef_init = True  # Take the initial coefficients from the previous iteration to warm-start the optimization.
    else:
        early_stopping = False
        average_coef = False
        eta_start_iter = 1
        coef_init = False
    early_stopping_score = 'v.PrecV.alpha'
    early_stopping_n_iter = 2500  # check each such iterations of SGD if the es_score is achieved.
    coef_init_cont_eta = False # continue with the same eta as ended in the previous iteration (speed up the convergence - less MW iterations).
    if private_params:
        random_blocks_num = 8 #'v.SampCompV.max_iter / 70000000'
    else:
        random_blocks_num = 1
    verbose = 300  # 0 - no information output, int number - output in each such iterations, even number - print also debug output
    cupy = False  # run sgd with cupy - not in use for now (somehow is slower then numpy)


class PrivateVC:
    if private_params:
        private = True  # True - private run, False - non-private furn
    else:
        private = False
    kappa = 'v.PrecV.alpha/(4*v.SampleV.chi_s)'  # 0.000007  # 0.1  # class MWTransferLearning. TODO: What the value need to be ?
    delta = 0.0001  # TODO: What the value need to be ? # delta of DF
    epsilon_s = 1.0
    epsilon_t = 0.5
    rr_laplace_b = '1 / (v.PrivateV.epsilon_t)'  # TODO: What the value need to be ? 2 / EPSILON # b parameter of laplace noise for randomized-response (sq-query)
    ss_noise_debug_mode = True  # print from _sgd_fast.pyx file

    # Not in use
    # ss_noise_max_iter = max((ss_noise_sigma ** 2) * sqrt(SampleVC.dim) / (PrecVC.alpha ** 2),
    #                         log(2 / PrecVC.beta) / PrecVC.alpha ** 2)
    # ss_noise_num_threads_rand = 0  # number of threads for SGD algorithm. Can slow down the program. use only when DIM is very very large.


# The real sample complexity of S
# MAX_SAMPLE_SIZE_S_MIN_ALPHA_TEMP = 'ceil((800 * (CHI_S + 1) * log2(1 / MIN_ALPHA) / MIN_ALPHA ** 2) * (P_DIM * log2(400 * P_DIM / MIN_ALPHA ** 3) + ' \
#                 'log2(8 / BETA))) '

# min(MAX_SAMPLE_SIZE, asymptotic of MAX_SAMPLE_SIZE_S_MIN_ALPHA_TEMP)
# The real sample complexity of S without constants and with the MIN_ALPHA
class SampCompVC:
    # For private: the noise in SGD algorithm:
    ss_noise_sigma = 'sqrt(20)*v.ModelV.lipschitz/v.ModelV.batch_size' #'max(sqrt(20)*v.ModelV.lipschitz/v.ModelV.batch_size, sqrt((16 * (1/v.PrivateV.epsilon_s) * v.ModelV.lipschitz**2 * log(1/v.PrivateV.delta) + 8 * v.ModelV.lipschitz**2)/log(v.PrivateV.kappa * v.SampCompV.sample_size_s)) / v.ModelV.batch_size)'

    #ss_noise_sigma = 'max(sqrt(20)*v.ModelV.lipschitz/v.ModelV.batch_size, sqrt((16 * (1/v.PrivateV.epsilon_s) * v.ModelV.lipschitz**2 * log(1/v.PrivateV.delta) + 8 * v.ModelV.lipschitz**2)/log(v.PrivateV.kappa * v.SampCompV.sample_size_s)) / v.ModelV.batch_size)'
    if private_params:
        eta_sgd = 'v.ModelV.diameter/v.SampCompV.ss_noise_sigma'
    else:
        eta_sgd = 'v.ModelV.diameter/v.ModelV.lipschitz'

    if private_params:
        # For private # TODO - to complete the formula (or no need because it is a run parameter)
        max_iter_raw = 'ceil(max(9 * v.ModelV.diameter**2 * (v.ModelV.lipschitz**2 + v.SampCompV.ss_noise_sigma**2 * v.SampleV.dim)/v.PrecV.alpha**2, 8 * v.ModelV.diameter**2 * v.ModelV.lipschitz**2 * log(2/v.PrecV.beta) / v.PrecV.alpha**2))'
        max_iter = 'v.SampCompV.max_iter_raw' #'min(10000, v.SampCompV.max_iter_raw)'    
    else:
        max_iter_raw = 'int(9 * v.ModelV.diameter**2 * v.ModelV.lipschitz**2/(4 * v.PrecV.alpha**2))'  # ~= 601,200,000,000,000 - It is value for non-private version from elad hazan page 55 (47)
        max_iter = 'min(10000, v.SampCompV.max_iter_raw)'



    if private_params:
        sample_size_s_raw = 'int(v.ModelV.batch_size * sqrt(52 * v.SampCompV.max_iter * v.MWalgV.mw_max_iter / (8 * v.PrivateV.kappa**2 * v.PrivateV.epsilon_s) * log(52 * v.SampCompV.max_iter * v.MWalgV.mw_max_iter / (8 * v.PrivateV.kappa**2 * v.PrivateV.epsilon_s))))'
        # The options:
        # 1) int: fixed number
        # 2) 'v.SampCompV.sample_size_s_raw': official bound (with ln(kn) in the denominator)
        # 3) 'n_div_ln' - according to the calculate_sample_size_s() function (calculates n/ln(kn))
        sample_size_s = 'n_div_ln'
    else:
        sample_size_s_raw = 'ceil((800 * (v.SampleV.chi_s + 1) * log2(1 / v.PrecV.alpha) / v.PrecV.alpha ** 2) * (v.MWalgV.p_dim * log2(400 * v.MWalgV.p_dim / v.PrecV.alpha ** 3) + log2(8 / v.PrecV.beta)))'
        # The options:
        # 1) int: fixed number
        # 2) 'v.SampCompV.sample_size_s_raw': official bound
        sample_size_s = 'v.SampCompV.sample_size_s_raw'

    # Used only for printing the upper bound
    sample_size_s_upper_bound = 'v.SampCompV.sample_size_s_raw'
    sample_size_t = 'ceil(2 * log(2*v.MWalgV.mw_max_iter / v.PrecV.beta) / ((v.PrivateV.epsilon_t ** 2) * (v.PrecV.tau ** 2)))'
    # In using only in the old version
    subsample_size = 'ceil((v.SampleV.dim + log(v.MWalgV.mw_max_iter / v.PrecV.beta)) / (v.PrecV.alpha + v.PrecV.alpha_h))'  # not relevant for CustomSVM

class RunVC:
    """
        :param sklearn_model:  if True convert cupy arrays to numpy before using function of scikit-learn
    """
    multiproc_num = 0  # number of parrallel runs. 0 is not parallel, n is number of repetitions to run in parallel
    # must to divide the rep_num
    num_rep = 1  # repeats of run for each combination of the parameters
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

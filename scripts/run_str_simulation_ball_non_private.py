#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import sys
from datetime import datetime
# noinspection PyUnresolvedReferences
from math import ceil, log2, exp, log, sqrt

# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
from scipy.stats import chi2

from SubsampleTestReweighBall.commands_args import parse_args

if __name__ == "__main__":
    from SubsampleTestReweighBall.params_non_private import SampleVC, PrecVC, ModelVC, PrivateVC, MWalgVC, SampCompVC, RunVC, PathVC
from SubsampleTestReweighBall.prepare_run import PrepareRun
from SubsampleTestReweighBall.experiments import run_experiments


class GlobalVariables:
    SampleV = None
    PrecV = None
    ModelV = None
    PrivateV = None
    MWalgV = None
    SampCompV = None
    RunV = None
    PathV = None


# For private: calculate n/ln(n) in calculation of sample_size_s in order to reduce the sample_size_s
# The formula is the left size of Theorem A.3.
# When sample_size_s == 'n_div_n'
def calculate_sample_size_s(v):
    sample_size_s_raw_short = int(v.ModelV.batch_size * sqrt(
        52 * v.SampCompV.max_iter * v.MWalgV.mw_max_iter / (8 * v.PrivateV.kappa ** 2 * v.PrivateV.epsilon_s)))
    sample_size_s = 100000
    while True:
        # print(sample_size_s, sample_size_s / log(sample_size_s), sample_size_s_raw)
        if sample_size_s / log(v.PrivateV.kappa * sample_size_s) < sample_size_s_raw_short:
            sample_size_s += 100000
        else:
            return sample_size_s


def validate_input_params(v):
    # If we suppose that the additional coordinate of the sample (against the intercept) is 1, and then we can to
    # enforce the intercept to be bounded by 1. But in the new version we don't require it.
    # max_std_k_t = 1 / sqrt(chi2.ppf(v.SampleV.frac_zero_label_t, v.SampleV.k))
    # if v.SampleV.std_k_t > max_std_k_t:
    #     raise IOError("The std_k_t value cannot be greater than {} (1/sqrt(threshold-of-0-label)".format(max_std_k_t))
    if v.ModelV.do_proj:
        if v.ModelV.intercept > 1:
            raise IOError("The intercept {} cannot be greater than 1".format(v.ModelV.intercept))

    if v.SampleV.k_coord[-1] > v.SampleV.dim:
        raise IOError('The k_coord: {} exceed the bound of the hypotheis: {}'.format(v.SampleV.dim, v.SampleV.k_coord))

    if v.PrivateV.private:
        valid_sigma = max(sqrt(20) * v.ModelV.lipschitz / v.ModelV.batch_size, sqrt((16 * (
                    1 / v.PrivateV.epsilon_s) * v.ModelV.lipschitz ** 2 * log(
            1 / v.PrivateV.delta) + 8 * v.ModelV.lipschitz ** 2) / log(
            v.PrivateV.kappa * v.SampCompV.sample_size_s)) / v.ModelV.batch_size)
        print('The minimum value of v.SampCompV.ss_noise_sigma is: {}'.format(v.SampCompV.ss_noise_sigma), flush=True)
        if v.SampCompV.ss_noise_sigma < valid_sigma:
            raise IOError(
                "The current value of v.SampCompV.ss_noise_sigma is: {}, but it need to be greater than {}".format(
                    v.SampCompV.ss_noise_sigma, valid_sigma))

    # For previous version that we bound the sample by 2*dimension under the bellow condition, but in current version
    # we find it empiricaly.
    # if v.SampCompV.sample_size_s >= v.PrecV.beta * (1.1432) ** v.SampleV.dim:
    #     raise IOError("The dimension too small. n={}, but it is need to be lower than beta * 1.1432^dim = {}, i.e. in these setting dim need to be greater than {}".format(
    #                 v.SampCompV.sample_size_s, v.PrecV.beta * (1.1432) ** v.SampleV.dim,
    #                 log(v.SampCompV.sample_size_s / v.PrecV.beta, 1.1432)))

    if v.PrivateV.private:
        # according the formula  that divide the regret to two part which of them bounded by alpha/2
        print("Expected error of SGD is: {}".format(
            max(3 * v.ModelV.diameter * sqrt(
                v.ModelV.lipschitz ** 2 + v.SampleV.dim * v.SampCompV.ss_noise_sigma ** 2) / (
                    sqrt(v.SampCompV.max_iter)), 2 * v.ModelV.lipschitz * v.ModelV.diameter * sqrt(
                8 * log(2 / v.PrecV.beta) / v.SampCompV.max_iter))), flush=True)


def evals(val, v):
    try:
        return eval(str(val)) # cannot do eval to numpy number, therefore need first to convert it to str
    except TypeError:
        return val
    except NameError:
        return val
    except SyntaxError:
        return val


def eval_str_params(v, print_value=False):
    if print_value:
        params_sorted = []
        print('The parameters are:', flush=True)
    for c in [v.PrecV, v.SampleV, v.MWalgV, v.ModelV, v.PrivateV, v.SampCompV, v.RunV]:
        for i in dir(c):
            if not i.startswith('_'):
                if print_value:
                    params_sorted.append([i, evals(getattr(c, i), v)])
                setattr(c, i, evals(getattr(c, i), v))
    if print_value:
        params_sorted.sort(key=lambda x: x[0])
        for param in params_sorted:
            print('\t', *param, flush=True)


def empirical_bound_on_examples(v):
    print('Start to calculate the empirical bound on the examples ...')
    norm2sq_threshs = np.zeros(10)
    for i in range(10):
        sample = np.random.normal(loc=v.SampleV.mean_s, scale=v.SampleV.std_s, size=(50000, v.SampleV.dim)) ** 4
        norm2sq_sample_s = np.sum(sample, axis=1)
        norm2sq_thresh = 0.0
        while True:
            frac_above_thresh = np.mean(norm2sq_sample_s > norm2sq_thresh)
            if frac_above_thresh < v.ModelV.norm2sq_percentage:
                norm2sq_threshs[i] = norm2sq_thresh
                break
            norm2sq_thresh += v.SampleV.std_s / 100
    v.ModelV.norm2sq = np.mean(norm2sq_threshs)
    print('The bound on norm2 squared of sample S (squared) is {} w.p. of {}'.format(v.ModelV.norm2sq,
                                                                                     1 - v.ModelV.norm2sq_percentage))


def run(args):
    print(
        "\nSimulation of SubsampleTestReweigh (STR) algorithm - Transfer learning with only SQ queries to target distribuiton.",
        flush=True)
    print('\nRun started ... ', flush=True)
    print('\nFull run parameters are written to log folder inside the output directory: {}\n'.format(args.output_dir),
          flush=True)

    v = GlobalVariables()
    v.SampleV = SampleVC()
    v.PrecV = PrecVC()
    v.ModelV = ModelVC()
    v.PrivateV = PrivateVC()
    v.MWalgV = MWalgVC()
    v.SampCompV = SampCompVC()
    v.RunV = RunVC()
    v.PathV = PathVC()

    date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    pr = PrepareRun(date, args, v=v)
    if v.ModelV.norm2sq == 'empirical':
        empirical_bound_on_examples(v)
    # run this several times, until all str in params.py file will be converted to values by eval
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    eval_str_params(v)
    if v.PrivateV.private and v.SampCompV.sample_size_s == 'n_div_ln':
        v.SampCompV.sample_size_s = calculate_sample_size_s(v)
    eval_str_params(v, print_value=True)  # Run again for printing the values to the bash screen
    pr.create_title()
    pr.prepare_out_dirs()
    pr.create_output_dirs()
    pr.write_parameters_file()
    validate_input_params(v)
    run_experiments(v)


if __name__ == "__main__":
    args = parse_args()
    run(args)

# Parameters for fast test - the algorithm convergence after one iteration:
# -------------------------------------------------------------------------
# --std-k-t 0.5

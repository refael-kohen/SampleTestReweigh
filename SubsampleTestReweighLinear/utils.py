import sys
# noinspection PyUnresolvedReferences
from logging import DEBUG, INFO, ERROR

import numpy as np  # numpy or cupy

try:  # if np is numpy
    np.seterr(divide='ignore')
    np.seterr(divide='warn')
except AttributeError:  # if np is cupy
    pass

# print all elements of long numpy array, and print it in one line
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


class Const:
    # penalties
    L2 = 'l2'
    L1 = 'l1'

    # models
    SVM = 'SVM'
    LR = 'LR'
    LP = 'LP'

    # plot and data_structure
    OUT_ITER_NUM = 'iterations_num'
    OUT_DIST_CHISQ = 'dist_chisq'
    OUT_DIST_KL = 'dist_kl'
    OUT_DIST_TV = 'dist_tv'
    OUT_LOSS_S = 'loss_S'
    OUT_LOSS_S_SUB = 'loss_sub_S'
    OUT_LOSS_T = 'loss_T'


def get_np(arr):
    """
    convert cupy array to numpy if is cupy, else return the input

    :param arr: cupy or numpy array
    :return: numpy array
    """
    try:
        return arr.get()
    except AttributeError:
        return arr

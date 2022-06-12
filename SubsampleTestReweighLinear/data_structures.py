# from settings import np
import os

from SubsampleTestReweighLinear.utils import *  # import np from utils


class DataStruct:
    def __init__(self) -> None:
        pass

    def print_template(self, file_name_pre, values, succeeded_rep, rep_counter, v):
        # in multiprocessing will be created different temp file for each repetition.
        # in non-parallel rep == ''
        repstr = str(rep_counter).zfill(5) + '_' if v.RunV.multiproc_num else ''
        file_name = os.path.join(v.PathV.output_path, 'sampSize_' + str(v.SampCompV.sample_size_s),
                                 '_'.join(
                                     [repstr + v.PathV.date, file_name_pre, str(v.SampCompV.sample_size_s)]) + '.txt')
        with open(file_name, 'a') as file_fh:
            np.savetxt(file_fh, np.append(values, succeeded_rep).reshape(1, -1), delimiter=',')
            file_fh.flush()
            os.fsync(file_fh.fileno())

    def print_data_repetition(self, succeeded_rep, iter_num, chisq_dist, kl_dist, tv_dist, loss_s, loss_s_sub,
                              loss_t, rep_counter, v):
        self.print_template(Const.OUT_ITER_NUM, iter_num, succeeded_rep, rep_counter, v)
        self.print_template(Const.OUT_DIST_CHISQ, chisq_dist, succeeded_rep, rep_counter, v)
        self.print_template(Const.OUT_DIST_KL, kl_dist, succeeded_rep, rep_counter, v)
        self.print_template(Const.OUT_DIST_TV, tv_dist, succeeded_rep, rep_counter, v)
        self.print_template(Const.OUT_LOSS_S, loss_s, succeeded_rep, rep_counter, v)
        self.print_template(Const.OUT_LOSS_S_SUB, loss_s_sub, succeeded_rep, rep_counter, v)
        self.print_template(Const.OUT_LOSS_T, loss_t, succeeded_rep, rep_counter, v)

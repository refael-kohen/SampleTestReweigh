import os
# noinspection PyUnresolvedReferences
from math import ceil, log2, exp, log, sqrt

# noinspection PyUnresolvedReferences
from scipy.stats import chi2

class PrepareRun:
    def __init__(self, date, args, v):
        self.date = date
        self.v = v
        self.fill_user_parameters(args)

    def fill_user_parameters(self, args):
        self.v.PathV.date = self.date
        # convert from linux to windos
        if '/' in args.output_dir:
            args.output_dir = os.path.join(*args.output_dir.split('/'))
        if "\\" in args.output_dir:
            args.output_dir = os.path.join(*args.output_dir.split("\\"))
        self.v.PathV.output_root = args.output_dir
        self.v.RunV.num_rep = args.num_rep
        self.v.RunV.multiproc_num = args.multiproc_num
        self.v.RunV.multiproc_num = min(self.v.RunV.num_rep, self.v.RunV.multiproc_num)
        if args.sample_size_s is not None:
            self.v.SampCompV.sample_size_s = args.sample_size_s
        if args.sample_size_t is not None:
            self.v.SampCompV.sample_size_t = args.sample_size_t
        if args.std_k_t is not None:
            self.v.SampleV.std_k_t = args.std_k_t
        if args.alpha is not None:
            self.v.PrecV.alpha = args.alpha
        if args.frac_zero_label_t is not None:
            self.v.SampleV.frac_zero_label_t = args.frac_zero_label_t
        if args.sgd_max_iter is not None:
            self.v.SampCompV.max_iter = args.sgd_max_iter
        if args.sgd_es_score is not None:
            self.v.ModelV.early_stopping = True
            self.v.ModelV.early_stopping_score = args.sgd_es_score
        if args.sgd_batch_size is not None:
            self.v.ModelV.batch_size = args.sgd_batch_size
        if args.sgd_reg_c is not None:
            self.v.ModelV.reg_c = args.sgd_reg_c
        if args.mw_max_iter is not None:
            self.v.MWalgV.mw_max_iter = args.mw_max_iter
        if args.mw_eta is not None:
            self.v.MWalgV.eta = args.mw_eta
        if args.dim is not None:
            self.v.SampleV.dim = args.dim
        if args.penalty is not None:
            self.v.ModelV.penalty = args.penalty
        self.v.PathV.title = args.title

    def create_title(self):
        if not self.v.PathV.title:
            self.v.PathV.title = 'sampleSizeS-{}_sampleSizeT-{}_alpha-{}_stdT-{}' \
                .format(self.v.SampCompV.sample_size_s, self.v.SampCompV.sample_size_t,
                        self.v.PrecV.alpha, self.v.SampleV.std_k_t)
            # self.v.PathV.title = 'std_t={}a={}_iter={}K_rep={}_eta={}' \
            #                      '_Kcoord={}_model={}_reg=1e{}'.format(self.v.SampleV.std_k_t, self.v.PrecV.alpha,
            #                                                    int(int(self.v.MWalgV.mw_max_iter) / 1000),
            #                                                    self.v.RunV.num_rep, self.v.ModelV.eta, self.v.SampleV.k,
            #                                                    ModelV.model_name,
            #                                                    len(str(ModelV.reg_c).strip('1')))

    def prepare_out_dirs(self):
        self.v.PathV.log_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title, 'logs')
        self.v.PathV.plot_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title, 'plots')
        self.v.PathV.output_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title, 'output')
        self.v.PathV.title_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title)

    def create_output_dirs(self):
        # os.makedirs(self.v.PathV.output_root, exist_ok=False)
        os.makedirs(self.v.PathV.title_path, exist_ok=True)
        os.makedirs(self.v.PathV.log_path, exist_ok=True)
        os.makedirs(self.v.PathV.plot_path, exist_ok=True)
        os.makedirs(self.v.PathV.output_path, exist_ok=True)
        # False - do not create new files in the previous run
        if self.v.PrivateV.private:
            os.makedirs(os.path.join(self.v.PathV.output_path, 'sgdIters_' + str(self.v.SampCompV.max_iter)),
                        exist_ok=True)  # False
        else:
            os.makedirs(os.path.join(self.v.PathV.output_path, 'sampSize_' + str(self.v.SampCompV.sample_size_s)),
                        exist_ok=True)  # False

    def write_parameters_file(self):
        users_parameter_file = os.path.join(self.v.PathV.log_path,
                                            '_'.join([self.v.PathV.date, str(self.v.SampCompV.sample_size_s),
                                                      'run_parameters.txt']))
        with open(users_parameter_file, 'w') as pf:
            pf.write("-------------------------------------\n")
            pf.write("Run parameters: {}:\n".format(self.date))
            pf.write("-------------------------------------\n\n")
            for c in [self.v.SampleV, self.v.PrecV, self.v.ModelV, self.v.PrivateV, self.v.MWalgV, self.v.SampCompV,
                      self.v.RunV, self.v.PathV]:
                pf.write("{}:\n".format(c.__class__.__name__))
                pf.write("---------\n")
                # pf.writelines(["{}: {}\n".format(i, j) for i, j in c.__dict__.items() if not i.startswith('_')])
                pf.writelines(["{}: {}\n".format(i, getattr(c, i)) for i in dir(c) if not i.startswith('_')])
                pf.write("\n")

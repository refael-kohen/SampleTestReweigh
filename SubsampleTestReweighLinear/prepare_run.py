import os


class PrepareRun:
    def __init__(self, date, args, v=None):
        self.date = date
        self.v = v

        self.fill_user_parameters(args, v)
        self.prepare_out_dirs(v)
        self.create_output_dirs(v)
        self.write_parameters_file(v)

    def fill_user_parameters(self, args, v):
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
        self.v.SampCompV.sample_size_s = args.sample_size_s
        self.v.SampCompV.sample_size_t = args.sample_size_t
        self.v.SampleV.std_k_t = args.std_k_t
        self.v.PrecV.alpha = args.alpha
        self.v.ModelV.penalty = args.penalty
        self.v.PathV.title = args.title
        if not args.title:
            self.v.PathV.title = 'sampleSizeS-{}_sampleSizeT-{}_alpha-{}_stdT-{}' \
                .format(self.v.SampCompV.sample_size_s, self.v.SampCompV.sample_size_t,
                        self.v.PrecV.alpha, self.v.SampleV.std_k_t)
            # self.v.PathV.title = 'std_t={}a={}_iter={}K_rep={}_eta={}' \
            #                      '_Kcoord={}_model={}_reg=1e{}'.format(self.v.SampleV.std_k_t, self.v.PrecV.alpha,
            #                                                    int(int(self.v.MWalgV.mw_max_iter) / 1000),
            #                                                    self.v.RunV.num_rep, self.v.PrecV.eta, self.v.SampleV.k,
            #                                                    self.v.ModelV.model_name,
            #                                                    len(str(self.v.ModelV.regularization_c).strip('1')))

    def prepare_out_dirs(self, v):
        self.v.PathV.log_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title, 'logs')
        self.v.PathV.plot_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title, 'plots')
        self.v.PathV.output_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title, 'output')
        self.v.PathV.title_path = os.path.join(self.v.PathV.output_root, self.v.PathV.title)

    def create_output_dirs(self, v):
        # os.makedirs(self.v.PathV.output_root, exist_ok=False)
        os.makedirs(self.v.PathV.title_path, exist_ok=True)
        os.makedirs(self.v.PathV.log_path, exist_ok=True)
        os.makedirs(self.v.PathV.plot_path, exist_ok=True)
        os.makedirs(self.v.PathV.output_path, exist_ok=True)
        # False - do not create new files in the previous run
        os.makedirs(os.path.join(self.v.PathV.output_path, 'sampSize_' + str(self.v.SampCompV.sample_size_s)),
                    exist_ok=False)

    def write_parameters_file(self, v):
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

import os
import shutil
from glob import glob

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SubsampleTestReweighBall.utils import Const  # Not !!! import np from utils (this script cannot work with cupy)

plt.rcParams.update({'font.size': 10})


class AggregateOutputFiles:
    def __init__(self, output_path, log_path, repetitions_num, private=False):
        self.output_path = output_path
        self.log_path = log_path
        self.repetitions_num = repetitions_num
        self.private = private
        self.fileName = 'sampSize' if not self.private else 'sgdIters'

    def aggregate_parallel_logs(self, sample_size_S):
        # in multiprocessing will be created different temp file for each repetition.
        log_names = glob(os.path.join(self.log_path, '*' + '_' + str(sample_size_S) + '.log'))[:self.repetitions_num]
        log_w_names = glob(os.path.join(self.log_path, '*' + '_' + str(sample_size_S) + '_weights.log'))[:self.repetitions_num]

        aggregated_log_name = os.path.join(self.log_path, '_'.join(['agg', str(sample_size_S)]) + '.log')
        aggregated_log_w_name = os.path.join(self.log_path,
                                             '_'.join(['agg', str(sample_size_S), 'weights']) + '.log')
        log_names = list(set(log_names) - set([aggregated_log_name]))  # remove old aggregated files from the list
        log_w_names = list(set(log_w_names) - set([aggregated_log_w_name]))  # remove old aggregated files from the list
        with open(aggregated_log_name, 'wb') as wfd:
            for f in log_names:
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)

        with open(aggregated_log_w_name, 'wb') as wfd:
            for f in log_w_names:
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)

    def aggregate_parallel_outputs_one_group(self, file_name_pre, sample_size_S):
        # in private mode sample_size_S is sgd_iterations
        # in multiprocessing will be created different temp file for each repetition.
        file_names = glob(os.path.join(self.output_path, self.fileName + '_' + str(sample_size_S), '*' + file_name_pre + '*'))[:self.repetitions_num]
        aggregated_file_name = os.path.join(self.output_path, self.fileName + '_' + str(sample_size_S),
                                            '_'.join(['agg', file_name_pre, str(sample_size_S)]) + '.txt')
        file_names = list(set(file_names) - set([aggregated_file_name]))  # remove old aggregated files from the list

        with open(aggregated_file_name, 'wb') as wfd:
            for f in file_names:
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)

    def aggregate_parallel_files(self, sample_size_S):
        # self.aggregate_parallel_logs(sample_size_S) # TODO: it takes a lot of time
        self.aggregate_parallel_outputs_one_group(Const.OUT_ITER_NUM, sample_size_S)
        self.aggregate_parallel_outputs_one_group(Const.OUT_DIST_CHISQ, sample_size_S)
        # self.aggregate_parallel_outputs_one_group(Const.OUT_DIST_KL, sample_size_S)
        self.aggregate_parallel_outputs_one_group(Const.OUT_DIST_TV, sample_size_S)
        self.aggregate_parallel_outputs_one_group(Const.OUT_LOSS_S, sample_size_S)
        self.aggregate_parallel_outputs_one_group(Const.OUT_LOSS_S_SUB, sample_size_S)
        self.aggregate_parallel_outputs_one_group(Const.OUT_LOSS_T, sample_size_S)


class Plot:
    def __init__(self, output_path, log_path, plot_path, sample_sizes, skip_iterations=50, limit_y_axis_std=-1,
                 private=False):
        self.output_path = output_path
        self.log_path = log_path
        self.plot_path = plot_path
        self.sample_sizes = sample_sizes
        self.skip_iterations = skip_iterations
        self.limit_y_axis_std = limit_y_axis_std
        # self.sample_sizes_str = '_'.join([str(int(ss/1000)) for ss in self.sample_sizes])
        self.sample_sizes_str = '-'.join([str(min(self.sample_sizes)), str(max(self.sample_sizes))])
        self.private = private
        self.title = 'sample size' if not self.private else 'SGD iterations'
        self.fileName = 'sampSize' if not self.private else 'sgdIters'

    #
    # def last_iter_col(self, df):
    #     # after insertion of self.title column in first column, and considering the last column - 'succeed'
    #     return len(df.columns)-3

    def get_file_list(self, file_name_pre):
        files_list = []
        for sample_size_S in self.sample_sizes:
            file_name = glob(
                    os.path.join(self.output_path, self.fileName + '_' + str(sample_size_S), 'agg*' + file_name_pre + '*'))
            files_list.extend(file_name)
        return files_list

    def concatenate_files_from_sample_sizes(self, file_name_pre):
        files_list = self.get_file_list(file_name_pre)
        df_list = []
        for filename, sample_size_S in zip(files_list, self.sample_sizes):
            df_one = pd.read_csv(filename, header=None)
            df_one.insert(0, self.title, sample_size_S)
            df_one.rename(columns={len(df_one.columns) - 2: 'succeed'}, inplace=True)
            df_one = df_one[df_one['succeed'] == 1.0].drop(['succeed'], axis=1).reset_index(drop=True)
            df_list.append(df_one)
        return pd.concat(df_list, ignore_index=False)

    def general_vs_iters(self, file_name_pre, col_name, label_graph=None):
        # add errorbar
        # https://stackoverflow.com/questions/35562556/plotting-error-bars-matplotlib-using-pandas-data-frame
        df = self.concatenate_files_from_sample_sizes(file_name_pre)
        df_grouped_mean = df.groupby(self.title).mean().T
        df_grouped_std = df.groupby(self.title).std().T
        # mean = df_grouped.mean()
        # std = df_grouped.std()
        fig, ax = plt.subplots()

        # for sample_size in self.sample_sizes:
        #     col_mean = df_grouped_mean[sample_size].dropna().iloc[::-self.skip_iterations].iloc[::-1].dropna()
        #     col_std = df_grouped_mean[sample_size].dropna().iloc[::-self.skip_iterations].iloc[::-1].dropna()
        #     plt.errorbar(x=range(1, len(col_mean)+1),
        #                  y=col_mean,
        #                  yerr=col_std)

        for sample_size in self.sample_sizes:
            # data = df_grouped_mean[sample_size].dropna().iloc[::-self.skip_iterations].iloc[::-1].dropna()
            # data_std = df_grouped_std[sample_size].dropna().iloc[::-self.skip_iterations].iloc[::-1].dropna()
            data = df_grouped_mean[sample_size].iloc[::-self.skip_iterations].iloc[::-1]
            data_std = df_grouped_std[sample_size].iloc[::-self.skip_iterations].iloc[::-1]
            mean = data.mean().mean()
            std = data.std().mean()
            if self.private:
                ax.errorbar(x=data.index, y=data, yerr=data_std, errorevery=20, label=sample_size, linewidth=0.7, capsize=3)
            else:
                ax = data.plot(label=sample_size, linewidth=0.7)
            if self.limit_y_axis_std >= 0:
                ax.set_ylim([mean - self.limit_y_axis_std * std, mean + self.limit_y_axis_std * std])
        ax.set_xlabel('Iterations')
        ax.set_ylabel(label_graph if label_graph else col_name)
        plt.legend(title=self.title)

        # plt.title('{} vs. sample size'.format(col_name))
        plt.savefig(os.path.join(self.plot_path,
                                 '{}_vs_iterations_{}.png'.format(col_name.replace(' ', '_'), self.sample_sizes_str)),
                    dpi=400)
        plt.close(fig)

    def last_not_nan(self, df):
        """
        #example of last_not_nan:
        df=pd.DataFrame({'a':[1,1,1],'b':[2,2,2],'c':[None,3,3],'d':[None,4,None],'e':[None,None,None]})
        idx_non_none = df.apply(lambda row: row.last_valid_index(),axis=1)
        df.lookup(idx_non_none.index, idx_non_none.values) # lookup is deprecated
        or
        df.reindex(idx_non_none.values, axis=1).to_numpy()[np.arange(len(df)), np.arange(len(df))]
        """
        idx_non_none = df.apply(lambda row: row.last_valid_index(), axis=1)
        return df.reindex(idx_non_none.values, axis=1).to_numpy()[np.arange(len(df)), np.arange(len(df))]

    def general_vs_sample_size(self, file_name_pre, file_out_name, label_graph=None):
        """
        Example in case that file_name_pre == 'Iterations'
        #iter_Vs_sample_Size example:
        df2 = pd.DataFrame({1:[8000,8000,8000], 2:[5,6,3], 'succeed':[1,1,0]})
        df2_succeed = df2[df2['succeed'] == 1.0].reset_index(drop=True)
        df1 = pd.DataFrame({1:[9000,9000,9000], 2:[7,5,2], 'succeed':[1,0,1]})
        df1_succeed = df1[df1['succeed'] == 1.0].reset_index(drop=True)
        df=pd.concat([df1_succeed,df2_succeed], ignore_index=False)
        df_sample_cols=df.pivot(columns=1, values=2)
        """
        df = self.concatenate_files_from_sample_sizes(file_name_pre)
        if file_out_name == 'Iterations':
            col_name = 0
        else:
            col_name = file_out_name
            df[col_name] = self.last_not_nan(df)
        # sample size became to be the columns
        df_sample_cols = df.pivot(columns=self.title, values=col_name)
        fig, ax = plt.subplots()
        mean = df_sample_cols.mean()
        std = df_sample_cols.std()
        ax.errorbar(x=df_sample_cols.columns, y=mean, yerr=std, marker='o', capsize=3)
        ax.set_xlabel(self.title)
        ax.set_ylabel(label_graph if label_graph else file_out_name)
        plt.xticks(df_sample_cols.columns)
        # df_sample_cols.boxplot()
        # plt.title('{} vs. sample size'.format(file_out_name))
        plt.xticks(rotation=90)
        fig.tight_layout()
        if self.private:
            plt.savefig(os.path.join(self.plot_path,
                                     '{}_vs_sgdIters_{}.png'.format(file_out_name.replace(' ', '_'),
                                                                       self.sample_sizes_str)), dpi=400)
        else:
            plt.savefig(os.path.join(self.plot_path,
                                     '{}_vs_sampleSizes_{}.png'.format(file_out_name.replace(' ', '_'),
                                                                       self.sample_sizes_str)), dpi=400)
        plt.close(fig)

    def plot_all(self):
        self.plot_iter_samp_size(Const.OUT_ITER_NUM, 'Iterations')
        self.plot_iter_samp_size(Const.OUT_LOSS_T, 'Loss T')
        self.plot_iter_samp_size(Const.OUT_LOSS_S, 'Loss S')
        # self.plot_iter_samp_size(Const.OUT_LOSS_S_SUB, 'Loss subsample S ')
        self.plot_iter_samp_size(Const.OUT_DIST_CHISQ, 'Chi-square distance',
                                 r'$\chi^2\left(\frac{\Pr_T}{\Pr_S}||w\right)$')
        self.plot_iter_samp_size(Const.OUT_DIST_KL, 'KL distance', r'$KL\left(\frac{\Pr_T}{\Pr_S}||w\right)$')
        self.plot_iter_samp_size(Const.OUT_DIST_TV, 'TV distance', r'$TV\left(\frac{\Pr_T}{\Pr_S}||w\right)$')

    def plot_iter_samp_size(self, file_name_pre, col_name, label_graph=None):
        if col_name != 'Iterations':
            self.general_vs_iters(file_name_pre, col_name, label_graph)
        self.general_vs_sample_size(file_name_pre, col_name, label_graph)

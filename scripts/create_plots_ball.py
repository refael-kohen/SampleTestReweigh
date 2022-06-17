#!/usr/bin/env python

import argparse
import os
import textwrap
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

from SubsampleTestReweighBall.plot import AggregateOutputFiles, Plot


class myArgparserFormater(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    """
    RawTextHelpFormatter: can break lines in the help text, but don't print default values
    ArgumentDefaultsHelpFormatter: print default values, but don't break lines in the help text
    """
    pass


def parse_args():
    help_txt = textwrap.dedent("""
    The script aggregates and plots outputs of several runs that made by run_str_simulation.py script 
    on different sample sizes from the source distribution.

    The plots are saved in the plots folder under the output folder.
    """)
    # "The previous steps:\n" \
    # "- The runs need to be by the main script of the package: run_str_simulation.py script.\n" \
    # "- All runs need to run with the same parameters except the --sample-size-s" \
    # "- The outputs of all results need to be under the same output directory (use the same --title parameter" \
    # "in all runs)\n\n" \
    # "Using this script:\n" \
    # "After the end of all runs, run this script with the command:\n" \
    # "python create_plots.py --input-dir OUTPUT_RUNS --sss-list 10000 20000 30000\n" \
    # "if the runs made in multiprocessing mode --multiproc_num > 0 in run_str_simulation.py script)\n" \
    # "need to add the parameter --run-parallel for aggregation the output from all processors.\n"

    parser = argparse.ArgumentParser(description=help_txt, formatter_class=myArgparserFormater)
    parser.add_argument("--input-dir", type=str, required=True,
                        help=
                        textwrap.dedent("""
                        Full path to the output folder which created by 
                        the --title parameter of the run_str_simulation.py script.
                        The subdirectories need to be separated by '/' (not '\\')
                        """))
    parser.add_argument("--sample-size-s-list", action="extend", nargs="+", type=int, required=True,
                        help='list of sample sizes from S of the different runs.')
    parser.add_argument("--num-rep", type=int, required=False, default=1,
                        help='Number of the repetition of the same run')
    parser.add_argument("--skip-aggregation", action='store_true',
                        help='Skip on aggregation step and only plot from aggregated files from previous run (or if there is only one repetition).')
    parser.add_argument("--skip-iterations", type=int, required=False, default=50,
                        help='How many iterations to plot in the figure of X vs iterations.')
    parser.add_argument("--limit-y-axis-std", type=float, required=False, default=-1.0,
                        help='Center the y value around mean +- limit-y-axis-std * std. Value of -1 indicates to not center.')
    parser.add_argument("--private", action='store_true',
                        help='It is a private run (the titles are different).')
    return parser.parse_args()


def create_plots(args):
    rep_num = args.num_rep
    root_dir = args.input_dir
    # convert from linux to windos
    if '/' in root_dir:
        root_dir = os.path.join(os.path.sep, *root_dir.split('/'))

    output_path = os.path.join(root_dir, 'output')
    log_path = os.path.join(root_dir, 'logs')
    plot_path = os.path.join(root_dir, 'plots')
    sample_sizes = args.sample_size_s_list

    if not args.skip_aggregation:
        agg = AggregateOutputFiles(output_path=output_path, log_path=log_path, repetitions_num=rep_num, private=args.private)
        for sample_size_S in sample_sizes:
            agg.aggregate_parallel_files(sample_size_S=sample_size_S)

    plot = Plot(output_path=output_path, log_path=log_path, plot_path=plot_path, sample_sizes=sample_sizes,
                skip_iterations=args.skip_iterations, limit_y_axis_std=args.limit_y_axis_std, private=args.private)
    plot.plot_all()


if __name__ == "__main__":
    args = parse_args()
    create_plots(args)

# # root = r'C:\Users\user\Documents\Msc-DS\thesis\article\MW_transfer\results_final\final_private\results_private_4alpha'
# min_sample_size = 10000
# max_sample_size = 90001
# step_sample_size = 10000
# rep_num = 50
# alpha = 0.01
# output_path = os.path.join(root, 'output')
# log_path = os.path.join(root, 'logs')
# plot_path = os.path.join(root, 'plots')
# sample_sizes = list(np.arange(min_sample_size, max_sample_size, step_sample_size))

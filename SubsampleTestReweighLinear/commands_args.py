import argparse
import textwrap
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter


class myArgparserFormater(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    """
    RawTextHelpFormatter: can break lines in the help text, but don't print default values
    ArgumentDefaultsHelpFormatter: print default values, but don't break lines in the help text
    """
    pass


def parse_args():
    help_txt = "Welcome to the simulation of SubsampleTestReweigh (STR) algorithm - Transfer learning with only SQ queries to target distribution.\n\n" \
               "The full list of parameters is located in the params.py script in the package SubsampleTestReweigh.\n\n" \
               "The basic command line is: python run_str_simulation.py"

    parser = argparse.ArgumentParser(description=help_txt, formatter_class=myArgparserFormater)

    RunVgroup = parser.add_argument_group("Run parameters")
    SampCompVgroup = parser.add_argument_group("Sample complexity parameters")
    ALgVgroup = parser.add_argument_group("Algorithm parameters")

    RunVgroup.add_argument("--output-dir", type=str, required=False, default='str-output-dir',
                           help=
                           textwrap.dedent("""
                           Full path to an existing output directory.
                           The subdirectories need to be separated by '/' (not '\\'))
                           """)
                           )
    RunVgroup.add_argument("--title", type=str, required=False, default=None,
                           help=
                           textwrap.dedent("""
                           Title of the run - the name of the sub-folder inside the folder.
                           Runs with different sample sizes can be the same title.
                           """)
                           )

    RunVgroup.add_argument("--num-rep", type=int, required=False, default=50,
                           help='Number of the repetition of the same analysis (each run with different random seed)')
    RunVgroup.add_argument("--multiproc-num", type=int, required=False, default=0,
                           help=
                           textwrap.dedent("""
                           Run with n processors. Set 0 for no multiprocessing run.
                           (in multiprocessing the screen log is mixed, you can see separeate 
                           log of each process in the logs folder - under the '--output-dir folder')
                           """)
                           )
    SampCompVgroup.add_argument("--sample-size-s", type=int, required=False, default=90000,
                                help='Sample size from the source distribution')
    SampCompVgroup.add_argument("--sample-size-t", type=int, required=False, default=50000,
                                help='Sample size from the target distribution')

    ALgVgroup.add_argument("--std-k-t", type=float, required=False, default=0.01,
                           help=
                           textwrap.dedent("""
                           Standard deviation of the target distribution on the k different coordinates
                           (for fast tests select 0.5 - then the run will finish after one iteration)
                           """)
                           )
    ALgVgroup.add_argument("--alpha", type=float, required=False, default=0.01,
                           help='Maximum error of the subsample learning algorithm (SVM) on the source distribution')
    ALgVgroup.add_argument("--penalty", type=str, required=False, default='l2', choices=['l2', 'l1'],
                           help='Penalty of the subsample learning algorithm(SVM)')
    return parser.parse_args()

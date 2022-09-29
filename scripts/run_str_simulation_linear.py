#!/usr/bin/env python

from datetime import datetime

from SubsampleTestReweighLinear.commands_args import parse_args

if __name__ == "__main__":
    from SubsampleTestReweighLinear.params import SampleVC, PrecVC, ModelVC, PrivateVC, MWalgVC, SampCompVC, RunVC, PathVC
from SubsampleTestReweighLinear.prepare_run import PrepareRun
from SubsampleTestReweighLinear.experiments import run_experiments


class GlobalVariables:
    SampleV = None
    PrecV = None
    ModelV = None
    PrivateV = None
    MWalgV = None
    SampCompV = None
    RunV = None
    PathV = None


def run(args):
    print(
        "\nSimulation of SubsampleTestReweigh (STR) algorithm - Transfer learning with only SQ queries to target distribuiton.")
    print('\nRun started ... ')
    print('\nFull run parameters are written to log folder inside the output directory: {}\n'.format(args.output_dir))

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

    PrepareRun(date, args, v=v)
    run_experiments(v)


if __name__ == "__main__":
    args = parse_args()
    run(args)

# Parameters for fast test - the algorithm convergence after one iteration:
# -------------------------------------------------------------------------
# --std-k-t 0.5

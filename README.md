## Table of Contents

- [Source code](#source-code)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Author](#Author)
- [License](#license)

## Source code

The source code and full documentation:
[source-code](https://github.com/refael-kohen/SampleTestReweigh)

## Background

This project is a simulation of Subsample-Test-Reweigh algorithm. The simulation and the results are described in the paper: "Transfer Learning In Differential Privacy's Hybrid-Model", accepted to International Conference of Machine Learning (ICML) - 2022:

The paper:
https://icml.cc/virtual/2022/spotlight/16176.

Poster: 
https://icml.cc/virtual/2022/poster/16175

For now, it can only be run on a sample of the distributions described in the paper, but in the near future, we intend to extend it to more general distributions.

## Install

Language programming: Python, Cython

This project has been packaged as Python package.

### Prerequest:
The package requires Cython>=0.29.26. If you want that Cython will be installed automatically in the installation process of the package you need to install "poetry" package of python:

```shell
$ conda install -c conda-forge poetry
```

```sh
$ cd SampleTestReweigh
$ python setup.py sdist bdist_wheel  
$ pip install dist\subsample-test-reweigh-0.0.2.tar.gz

OR

$ cd SampleTestReweigh
$ python setup.py sdist bdist_wheel
$ pip install dist\subsample_test_reweigh-0.0.2-cp38-cp38-win_amd64.whl
```
It is recommended to use **virtualenv** or **conda** to create a clean python environment.

The project can run on python>=3.8 in the Windows and Linux operating system.

The following dependencies are required and will be installed automatically with the above command:

NumPy, Scipy, Pandas, Scikit-Learn, Matplotlib.

#### Test the installation:

On linux, run in the python environment the command:
```sh
$ run_str_simulation_linear.py --help  
```

On windows see later in [windows](#windows-environment) 

## Usage

The following commands are appropriate to Linux, for windows there is a little change see [windows](#windows-environment).

#### **Step 1:** running the "Linear" simulation on different sample sizes from the source distribution:

The main script called: run_str_simulation_linear.py
	
The script runs the simulation with one set of parameters. In the simulation described in the paper, we run it several times, each with a different sample size from the source distribution. 

Following the command for running 50 repetitions of the simulation (each with a different seed of the random numbers generator), with a sample size of 90K examples from the source distribution. We run it in parallel on 25 processors (so 25 repetitions run first and after them another group). 

```sh
$ run_str_simulation_linear.py --output-dir str-output-dir --title paperExample --multiproc 25 --num-rep 50 --sample-size-s 90000
```

In a similar way, we run the following similar command for the --sample-size-s of 100K and so on until 140K.

Notice that we put the results under the same --output-dir folder and the same --title, in order to enable us to aggregate and plot the results from all runs.

```sh
$ run_str_simulation_linear.py --output-dir str-output-dir --title paperExample --multiproc 25 --num-rep 50 --sample-size-s 100000
```

The running time of one repetition is ~9-14 hours (more time for a small sample size).

Full help on the parameters of the script you can achieve with the command:

```sh
$ run_str_simulation_linear.py --help  
```

The output is: [run-str-simulation-help](str-output-dir/run_str_sumulation_cmd.txt)

#### **Step 2:** agreggate and plot the results.

After the all runs (with the different sample sizes) ended, we run another script for aggregating the output files and creating plots:

```sh
$ create_plots_linear.py --input-dir str-output-dir/paperExample --sample-size-s-list 90000 100000 110000 120000 130000 140000 --num-rep 50 --run-parallel
```
which means that the script needs to aggregate outputs of 50 repetitions of each of the runs in the --sample-size-s-list list. For proper aggregation, we also must indicate that the runs are made in parallel by adding the --run-parallel flag.

The plots will be located in str-output-dir/paperExample/plots folder.

Full help on the parameters of the script you can achieve with the command:
```sh
$ create_plots_linear.py --help  
```
The output is: [create-plots-help](str-output-dir/create_plots_cmd.txt)

### Run the "Ball" simulation

In the similar way you can run the "ball" simulation in private and non-private modes:

Non-private mode:
```sh
$ python run_str_simulation_cycle_non_private.py --output-dir str-output-dir --multiproc 50 --num-rep 50 --sample-size-s 50000 --sample-size-t 50000 --title ball-non-private --std-k-t 0.4
```
In a similar way, we run the following similar command for the --sample-size-s of 10K 25K 50K 75K 100K 125K.

The running time of one repetiton is ~3 hours and takes ~100M of RAM memory.

Plot graphs:
```sh
$ python create_plots_cycle.py --input-dir str-output-dir/ball-non-private --sample-size-s-list 10000 25000 50000 75000 100000 125000 --num-rep 50
```

Private mode:
```sh
$ python run_str_simulation_cycle_private.py --output-dir str-output-dir --multiproc 0 --num-rep 1 --title ball-private --std-k-t 0.07 --mw-max-iter 250 --sgd-max-iter 50000000 --alpha 0.08
$ python run_str_simulation_cycle_private.py --output-dir str-output-dir --multiproc 0 --num-rep 1 --title ball-private --std-k-t 0.07 --mw-max-iter 200 --sgd-max-iter 75000000 --alpha 0.08
$ python run_str_simulation_cycle_private.py --output-dir str-output-dir --multiproc 0 --num-rep 1 --title ball-private --std-k-t 0.07 --mw-max-iter 150 --sgd-max-iter 100000000 --alpha 0.08
```
Repeat the above commands 50 times (for 50 repetitions).

The private mode takes ~55G of memory and the running time of one repetiton is ~30 hours.

Plot graphs:
```sh
$ python create_plots_cycle.py --input-dir ball-private --sample-size-s-list  --num-rep 50 --private
```

### Toy example of run:

In order to illustrate the running of the script, you can run a toy example of the linear example (Each of the following commands takes ~5-10 minutes and can be run on any computer with OS of Linux or Windows):

In the toy example, we use the parameter: --std-k-t 0.04, which means that the standard deviation of the K different coordinates of the target distribution is 0.04 instead of 0.01 in our real example in the paper. In this example, the algorithm converges very fast because no the distributions are not very far from each 
other.

```sh
$ run_str_simulation_linear.py --output-dir str-output-dir --title toyRunForTest --std-k-t 0.04 --multiproc 10 --num-rep 10 --sample-size-s 50000
$ run_str_simulation_linear.py --output-dir str-output-dir --title toyRunForTest --std-k-t 0.04 --multiproc 10 --num-rep 10 --sample-size-s 75000
$ run_str_simulation_linear.py --output-dir str-output-dir --title toyRunForTest --std-k-t 0.04 --multiproc 10 --num-rep 10 --sample-size-s 100000
```
```sh
$ create_plots_linear.py --input-dir str-output-dir/toyRunForTest --sample-size-s-list 50000 75000 100000 --num-rep 10 --run-parallel --skip-iterations 1  
```

#### Example of output of the toy run

Example of the output files of the toy example you can find under 
[str-output-dir](str-output-dir) folder , and the plots are located under the [plots](str-output-dir/toyRunForTest/plots) folder.

### Windows environment

In Windows the location of the script is unknown, so you need to write the full path of the script and run with the word "python" before the command:

To find the location of the run_str_simulation_linear.py script, type the following command:

```sh
python -c "import os, sys; print(os.path.dirname(sys.executable)+'\Scripts')"
```
Given the output of the above command is:
C:\Users\user\\.conda\envs\ds\Scripts

The run commands need to be: 
```sh
python C:\Users\user\.conda\envs\ds\Scripts\run_str_simulation_linear.py .....
python C:\Users\user\.conda\envs\ds\Scripts\create_plots_linear.py .....
```

### Advanced parameters
A full list of parameters can be found in the following files (The parameters in the command line override the values in these files):

[params-linear.py](SubsampleTestReweighLinear/params.py)  

[params-ball-non-private.py](SubsampleTestReweighBall/params_non_private.py) 

[params-ball-private.py](SubsampleTestReweighBall/params_private.py) 


## Author

The code of the simulation has been written by Refael Kohen: refael.kohen@gmail.com 

## License

[BSD 3-Clause](LICENSE)


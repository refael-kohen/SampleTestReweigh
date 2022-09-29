from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
import os

with open("README.md", 'r') as f:
    long_description = f.read()

# create setup.py for package with cython
# https://levelup.gitconnected.com/how-to-deploy-a-cython-package-to-pypi-8217a6581f09

requires = ['numpy>=1.22.0', 'scipy>=1.6.3', 'pandas>=1.2.0', 'scikit-learn==1.0.2', 'matplotlib>=3.3.3']

setup(
    name="subsample-test-reweigh",
    version="0.0.2",
    packages=find_packages(),
    author="Refael Kohen",
    author_email="refael.kohen@gmail.com",
    description="SubsampleTestReweigh algorithm for transfer learning",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="https://github.com/AnonymousICML2022/SampleTestReweigh",
    url="https://github.com/refael-kohen/SampleTestReweigh",
    python_requires='>=3.8',
    install_requires=requires,  # Install dependencies
    scripts=[
        'scripts/run_str_simulation_cycle_non_private.py',
        'scripts/run_str_simulation_cycle_private.py',
        'scripts/create_plots_cycle.py',
        'scripts/run_str_simulation_linear.py',
        'scripts/create_plots_linear.py',
    ],

    include_dirs=[numpy.get_include()],
    ext_modules=cythonize([os.path.join("SubsampleTestReweighBall", "private_svm_sgd", "_sgd_fast_custom.pyx"),
                           os.path.join("SubsampleTestReweighBall", "private_svm_sgd", "_weight_vector_custom.pyx")]),
    zip_safe=False,

)

# for inatallation with pyproject.toml - install cython automatically
#conda install -c conda-forge poetry

# Run tests:
# open anaconda prompt window
# conda activate ds (or any environment with sklearn version 1.0.2)
# cd "C:\Users\user\Documents\Msc-DS\Thesis\article\sample-test-reweigh"
# python setup.py build_ext --inplace
# cd tests
# python test_sgd_classifier_custom.py
# python setup.py build_ext --inplace && python run_str_simulation.py --output-dir str-output-dir --title paperExample --multiproc 0 --num-rep 1 --std-k-t 0.4 --frac-zero-label-t 0.3 --sgd-max-iter 10000 --sgd-batch-size 150 --mw-eta 0.03 --sgd-reg-c 1 --sample-size-s 150000 --alpha 0.002 --sgd-es-score 0.002 --title frac=

# python setup.py build_ext --inplace && python run_str_simulation.py --output-dir str-output-dir --multiproc 0 --num-rep 1 --std-k-t 0.6 --frac-zero-label-t 0.3  --title private

# Run with installation
# delete dist, build directories
# del /q dist build
## wheel contains the pyd files (this command creates them), sdist don't contain them
## but these files are created in installation time by cython (from pyx files), for
## enabling compilation in installation time you must including "scikit-learn==1.0.2" in
## pyproject.toml file, rather than it not could find sklearn package in the installation
# python setup.py sdist bdist_wheel
# pip uninstall subsample_test_reweigh
# pip install dist\subsample-test-reweigh-0.0.2.tar.gz
## or
# pip install dist\subsample_test_reweigh-0.0.2-cp38-cp38-win_amd64.whl

# one line commnad:

# del /q dist build && python setup.py sdist bdist_wheel && pip uninstall -y subsample_test_reweigh && pip install dist\subsample_test_reweigh-0.0.2-cp38-cp38-win_amd64.whl && python C:\Users\user\.conda\envs\ds\Scripts\run_str_simulation.py --output-dir str-output-dir --title paperExample --multiproc 0 --num-rep 1 --std-k-t 0.4 --frac-zero-label-t 0.3 --sgd-max-iter 10000 --sgd-batch-size 150 --mw-eta 0.03 --sgd-reg-c 1 --sample-size-s 150000 --alpha 0.002 --sgd-es-score 0.002 --title frac=



# create plots:
# create_plots.py --input-dir str-output-dir/paperExample --sample-size-s-list 90000 100000 110000 120000 130000 140000 --num-rep 50 --run-parallel

This repository contains code for reproducing results found in [Confidence-Ranked Reconstruction of Census Microdata from Published Statistics](https://arxiv.org/abs/2211.03128)

# Setup

We will setting up our codebase by first creating an Anaconda environment. 
Please note that our codebase currently supports Python **3.7-3.8**.
````
conda create -n raprank python=3.8
conda activate raprank-test
pip install --upgrade pip
````

Next, we install PyTorch **1.12.0** (other versions of PyTorch have not yet been tested) by following these 
[instructions](https://pytorch.org/get-started/previous-versions/) according to your system specifications. 
For example, the following command installs PyTorch with CUDA 11.6 support on Linux via pip.
````
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
````

Our experiments make use of the codebase found in [here](https://github.com/terranceliu/dp-query-release). For this tutorial, 
we will simply clone this repository directly into our root directory and run setuptools.
````
git clone git@github.com:terranceliu/dp-query-release.git
cd dp-query-release
pip install -e .
````

Note that the dp-query-release codebase provides code for downloading and preprocessing a variety of datasets, including
the ones used for our experiments here. Therefore, we will create a symbolic link in our root directory to the `datasets` directory. 
````
ln -s $PWD/datasets ../datasets
````

All done! You may now return to the root directory (`cd ..`).

# Data

### Census

We run our Census data experiments using the Privacy-Protected Microdata File (PPMF) from the 
[2020-05-27 update](https://www2.census.gov/programs-surveys/decennial/2020/program-management/data-product-planning/2010-demonstration-data-products/01-Redistricting_File--PL_94-171/2020-05-27_ppmf/). 
More information about this demonstration data program can be found 
[here](https://www.census.gov/programs-surveys/decennial-census/decade/2020/planning-management/process/disclosure-avoidance/2020-das-development.html).
Please use the following commands to download the raw file (note that the size of this file is about 15 GB).
````
mkdir -p datasets/raw/ppmf/
cd datasets/raw/ppmf/
curl -o 2020-05-27-ppmf.csv https://www2.census.gov/programs-surveys/decennial/2020/program-management/data-product-planning/2010-demonstration-data-products/01-Redistricting_File--PL_94-171/2020-05-27_ppmf/2020-05-27-ppmf.csv
cd ../../..
````

To make our data preprocessing more memory efficient later on, we will first split the PPMF file into separate files 
for each state (using FIPS codes). Please note that this script takes several minutes to run per state (several hours in total).
````
mkdir -p datasets/raw/ppmf/by_state
cd dp-query-release
./examples/data_preprocessing/scripts/get_all_states.sh
````

Then from the same directory (`dp-query-release`), run
````
./examples/data_preprocessing/scripts/preprocess_tracts.sh -s 0
````
to obtain files for tract-level experiments, and
````
./examples/data_preprocessing/scripts/preprocess_blocks.sh
````
to obtain files for block-level experiments.

Note that for tract-level experiments, we randomly select a tract from each state by fixing the random seed. 
To create files for other random sets of tracts not used for our paper, you can change this seed, 
which is set as 0 for our experiments through the command line argument `-s 0`. 

Blocks are chosen according to their size relative to maximum block size in each state. This selection logic is 
hard-coded into `dp-query-release/examples/data_preprocessing/preprocess_ppmf_blocks.py`.

### ACS

We run experiments on ACS data derived from the [Folktables](https://github.com/zykls/folktables) package. 
To create the necessary data files, please run the following commands from the root directory of this repository.
````
cd dp-query-release
python examples/data_preprocessing/preprocess_folktables.py \
--tasks employment coverage mobility \
--states CA NY TX FL PA;
````
You may change which states or tasks you wish to create datasets for by using the `--tasks` and `--states` arguments.

# Execution

All experiments presented in our paper can be reproduced using the bash scripts found in the `scripts` directory.
If you would like understand more about the arguments used in these scripts, please check 
[arguments.py](https://github.com/terranceliu/rap-rank-reconstruction/blob/master/utils/arguments.py).

For each set of experiments, you will find the following files:
* `train.sh`: Fits 100 synthetic data generators (RAP) to the corresponding set of queries.
* `eval.sh`: Evaluates the set candidates outputted by the trained generators and saves results to the `results` directory.

which are organized into the following experiment identifiers (referenced below as `EXPERIMENT_NAME`):
* `ppmf_tracts`: Census (PPMF) tract-level experiments.
* `ppmf_tracts_ib`: Census (PPMF) tract-level experiments with the BLOCK attribute is excluded.
* `ppmf_blocks`: Census (PPMF) block-level experiments.
* `folktables`: ACS (Folktables) state-level experiments.


### RAP-RANK (initialized randomly)

To reproduce our results, simply run following scripts in `scripts/init_random` directory.
````
./scripts/init_random/<EXPERIMENT_NAME>/train.sh
./scripts/init_random/<EXPERIMENT_NAME>/eval.sh
````

where `EXPERIMENT_NAME` can be selected from `ppmf_tracts`, `ppmf_tracts_ib`, `ppmf_blocks`, and `folktables`.

Note that for `ppmf_tracts` and `ppmf_tracts_ib`, you must specify the seed used to generate the data files above.
In our experiments, this seed is set to 0, so you would run each bash script above with the argument `-s 0`:
````
./scripts/init_random/ppmf_tracts/train.sh -s 0
./scripts/init_random/ppmf_tracts/eval.sh -s 0
````

### RAP-RANK (initialized to the baseline distribution)

To reproduce our experiments in which RAP-RANK is initialized to the baseline distribution for tract- and block-level experiments,
you can instead run the scripts found in the `scripts/init_baseline` directory. For example for tract-level experiments, 
use the following:
````
./scripts/init_baseline/ppmf_tracts/train.sh -s 0
./scripts/init_baseline/ppmf_tracts/eval.sh -s 0
````

### Baselines

Scripts running our baselines can be found in `scripts/baselines`, 
where you will find the file `<EXPERIMENT_NAME>.sh` for each set of experiments. For example, you can run
````
./scripts/baselines/ppmf_tracts.sh -s 0
````
to generate baseline results for tract-level experiments, and
````
./scripts/baselines/ppmf_blocks.sh
````
for block-level experiments.

# Plotting

We have included some files in the `plotting` directory for producing plots similar to those found in our paper.

Once all experiments are completed, first run 
````
python plotting/plotting/concat_results.py
````
to collect all results files into a few files that will be used by the plotting scripts.

The run from the `plotting` directory, run
````
python prepare_files_folktables.py
python plot_folktables.py
````
and
````
python prepare_files_ppmf.py --experiment_type <EXPERIMENT_TYPE>
python plot_ppmf.py --experiment_type <EXPERIMENT_TYPE>
````
where `EXPERIMENT_NAME` can be selected from `tracts`, `tracts_ib`, and `ppmf_blocks`. 

Note that `prepare_files_folktables.py` and `prepare_files_ppmf.py` save their outputs to `plotting/results`
and do not need to be run again (unless you wish to overwrite their outputs) 
before running `plot_folktables.py` and `plot_ppmf.py`

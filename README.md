# Setup

We will begin our setup by first creating an Anaconda environment (Note that our codebase currently supports Python **3.8**.) 
````
conda create -n watchdog-test python=3.8
conda activate watchdog-test
pip install --upgrade pip
````

Next, we install PyTorch **1.12.0** (other versions of PyTorch have not yet been tested) by following these 
[instructions](https://pytorch.org/get-started/previous-versions/) according to your system specifications. 
For example, the following command installs PyTorch with CUDA 11.6 support on Linux via pip.
````
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
````

Our experiments make use of the codebase found in git@github.com:terranceliu/dp-query-release.git. For this tutorial, 
we will simply clone this repository directly into this directory and run setuptools.
````
git clone git@github.com:terranceliu/dp-query-release.git
cd dp-query-release
pip install -e .
cd ..
````

Note that the dp-query-release codebase provides code for downloading and preprocessing a variety of datasets, including
the ones used for our experiments here. Therefore, we will create a symbolic link in our root directory to the `datasets` directory. 
````
ln -s $PWD/datasets ../datasets
````

All done!

# Execution

All our experiments can be reproduced using the bash scripts found in the `scripts` directory. For each set of experiments,
you will find the following sets of files:
* `train.sh`: Fits 100 synthetic data generators (RAP) to the corresponding set of queries.
* `eval.sh`: Evaluates the set candidates outputted by the trained generators and outputs results to the `results` directory.
* `baseline.sh`: Runs the baselines relevant for each dataset and outputs results to the `results` directory.

## Census (Tract-level)



## ACS

We run experiments on ACS data derived from the Folktables dataset [link]. To create the necessary data files,
please run the following commands from the root directory of this repository.
````
cd dp-query-release
python examples/data_preprocessing/preprocess_folktables.py \
--tasks employment coverage mobility \
--states CA NY TX FL PA;
cd ..
````

To reproduce our results, simply run following commands.
````
./scripts/folktables/train.sh
./scripts/folktables/eval.sh
````
and
````
./scripts/folktables/baseline.sh
````
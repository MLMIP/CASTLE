# CASTLE
***
## Improving mutation pathogenicity prediction of metal-binding sites in proteins with a panoramic attention mechanism
***

CASTLE is a novel framework based on depth map panoramic attention weaving learning, which significantly improves the mutation pathogenicity prediction of metal-binding sites. The core of CASTLE effectiveness lies in the efficient induction of metalloprotein structures and the integration of multidimensional features.

## Step 1: Clone the GitHub repository

```commandline
git clone https://github.com/MLMIP/CASTLE.git
cd CASTLE
```

## Step 2: Build required dependencies
It is recommended to use [Anaconda](https://www.anaconda.com/download#downloads) to install PyTorch, PyTorch Geometrics 
and other required Python libraries. Executing the below command will automatically install the Anaconda virtual 
environment. Upon completion, a virtual environment named "CASTLE" will be created. (dgl:1.1.2+cu118)
```commandline
source install.sh
```

## Step 3: Download required software
The download methods for various software packages are provided below. After downloading, you can install it directly 
according to the official tutorial. It should be noted that after installing the following software, the paths in 
gen_features.py need to be modified to the corresponding paths of the installed software.

1. **Naccess**\
Download from http://www.bioinf.manchester.ac.uk/naccess/
2. **Hbplus**\
Download from https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/


## Step 4: Running CASTLE
Activate the installed CASTLE virtual environment and ensure that the current working directory is CASTLE.
```commandline
conda activate CASTLE
```
Then, you can use the following command to batch predict the effect of mutations on protein stability.
```commandline
python predict.py -i /path/to/where/input/file -o /path/to/where/output/file -d /path/to/where/all/features/is/stored
```
For example:
```commandline
python predict.py -i ./predict_save_data/predict.xlsx -o ./predict_save_data/CASTLE_predict_result.xlsx -d ./predict_save_data
```
Where the input file is a given file, with each line representing a specific mutation in the format 
`PDB Metal FromAA ToAA pdbpos`, such as `1s1c MG F L 39`.

## other
We provide all curated samples of metal binding site mutations, the code for generating sample features, and the prediction scores of all methods on the test set. To train your own data, please generate sample features and then use files such as fivefold_crossvalid.py. The hyperparameter settings for all datasets are stored in the config.yaml file.

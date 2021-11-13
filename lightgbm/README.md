# LightGBM benchmarks

Fitting gradient boosted trees models using LightGBM.

## Environment setup

Generally following the instructions here:
https://towardsdatascience.com/install-xgboost-and-lightgbm-on-apple-m1-macs-cb75180a2dda

First install a few libraries with brew:
```bash
brew install cmake libomp gcc
```

Then install required packages.
Recommended to install in a new conda environment.
For example:

```bash
conda create -n benchmark-lightgbm python=3.8
conda activate benchmark-lightgbm
conda install numpy scipy scikit-learn
conda install lightgbm
```

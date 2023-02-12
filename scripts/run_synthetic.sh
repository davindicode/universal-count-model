#!/bin/bash


### fitting ###
cd ./fit/


# regression models
python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP-exp --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood hNB-exp --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping ffnn-50-50-100 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0


# progressively capturing single neuron variability and noise correlations
python3 models.py --data_type modIP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP-exp --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type modIP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type modIP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 5000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-16 --x_mode hd --z_mode R1 --lr 1e-2 --jitter 1e-5 --gpu 0


# circular latent variable models
python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP-exp --mapping svgp-8 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood hNB-exp --mapping svgp-8 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-4 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping ffnn-50-50-100 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0


### analysis ###
cd ../analysis/

python3 hCMP_analysis.py
python3 modIP_analysis.py


### plots ###
cd ../plots/

python3 schematic_plots.py
python3 synthetic_plots.py
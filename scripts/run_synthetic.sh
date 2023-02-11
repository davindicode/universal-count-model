#!/bin/bash

# by default, attempts to launch commands on GPU 0


# regression models
python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP-exp --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood hNB-exp --mapping svgp-8 --x_mode hd --lr 5e-3 --jitter 1e-4 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping ffnn-50-50-100 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0


# progressively capturing single neuron variability and noise correlations
python3 models.py --data_type IP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type IP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

#python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 8 --datatype 1 --ncvx 2 --lr 1e-2 --batchsize 5000
python3 models.py --data_type IP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 5000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --z_mode R1 --lr 1e-2 --jitter 1e-5 --gpu 0


# circular latent variable models
python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping ffnn --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0




python3 models.py --data_type hCMP1 --checkpoint_dir ../checkpoint/ --data_path ../../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood NB-exp --mapping svgp-6 --x_mode hd --lr 5e-3 --jitter 1e-4 --gpu 0

# analysis
python3 hCMP_analysis.py
python3 modIP_analysis.py


# plots
python3 schematic_plots.py
python3 synthetic_plots.py
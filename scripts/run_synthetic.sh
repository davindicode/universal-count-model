#!/bin/bash


# modes_tot = [('GP', 'IP', 'hd', 8, 'exp', 1, [], False, 10, False, 'ew'), # 1
#              ('GP', 'hNB', 'hd', 8, 'exp', 1, [], False, 10, False, 'ew'), 
#              ('GP', 'U', 'hd', 8, 'identity', 3, [], False, 10, False, 'ew'), 
#              ('ANN', 'U', 'hd', 8, 'identity', 3, [], False, 10, False, 'ew'), 
#              ('GP', 'IP', 'T1', 8, 'exp', 1, [0], False, 10, False, 'ew'), # 5
#              ('GP', 'hNB', 'T1', 8, 'exp', 1, [0], False, 10, False, 'ew'), 
#              ('GP', 'U', 'T1', 8, 'identity', 3, [0], False, 10, False, 'ew'), 
#              ('ANN', 'U', 'T1', 8, 'identity', 3, [0], False, 10, False, 'ew'), 
#              ('GP', 'U', 'hdxR1', 16, 'identity', 3, [1], False, 10, False, 'ew')] # 9


# regression models
# python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 0 1 2 3 --datatype 0 --ncvx 2 --lr 1e-2
python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP-exp --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood hNB-exp --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0 &

python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP1 --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping ffnn-50-50-100 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0



# progressively capturing single neuron variability and noise correlations
#python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 0 2 --datatype 1 --ncvx 2 --lr 1e-2
python3 models.py --data_type IP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood IP --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type IP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --lr 1e-2 --jitter 1e-5 --gpu 0

#python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 8 --datatype 1 --ncvx 2 --lr 1e-2 --batchsize 5000
python3 models.py --data_type IP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 5000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --x_mode hd --z_mode R1 --lr 1e-2 --jitter 1e-5 --gpu 0


# latent variable models
# python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 4 5 6 7 --datatype 0 --ncvx 3 --lr 1e-2 --lr_2 1e-3
python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping svgp-8 --z_mode T1 --lr 1e-2 --jitter 1e-5 --gpu 0

python3 models.py --data_type hCMP --checkpoint_dir ./checkpoint/ --data_path ../data/ --cv_folds 10 --cv -1 2 5 8 --seeds 123 1234 12345 --batch_size 10000 --max_epochs 3000 --bin_size 1 --likelihood U-el-3 --mapping ffnn --z_mode T1 --lr 1e-2 --lr_2 1e-3 --jitter 1e-5 --gpu 0
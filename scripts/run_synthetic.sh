#!/bin/bash




# regression models
python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 0 1 2 3 --datatype 0 --ncvx 2 --lr 1e-2


# latent variable models
python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 4 5 6 7 --datatype 0 --ncvx 3 --lr 1e-2 --lr_2 1e-3


# progressively capturing single neuron variability and noise correlations
python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 0 2 --datatype 1 --ncvx 2 --lr 1e-2
python3 validation.py --cv -1 2 5 8 --gpu 0 --modes 8 --datatype 1 --ncvx 2 --lr 1e-2 --batchsize 5000
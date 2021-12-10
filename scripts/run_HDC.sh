#!/bin/bash




# regression with different likelihoods
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 0 1 4 --ncvx 2 --lr 1e-2 --binsize 40


# regression with smaller subset of input covariates
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 2 3 --ncvx 2 --lr 1e-2 --binsize 40


# regression with different time bin sizes
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 4 --ncvx 2 --lr 1e-2 --binsize 20
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 4 --ncvx 2 --lr 1e-2 --binsize 100
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 4 --ncvx 2 --lr 1e-2 --binsize 200
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 4 --ncvx 2 --lr 1e-2 --binsize 500


# joint latent-observed models
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 5 6 7 8 --ncvx 3 --lr 1e-2 --lr_2 1e-3 --binsize 40


# latent variable models
python3 HDC.py --cv -1 1 2 3 5 6 8 --gpu 0 --modes 9 10 11 --ncvx 3 --lr 3e-2 --lr_2 5e-3 --binsize 100
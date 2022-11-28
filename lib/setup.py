import os

from setuptools import find_packages, setup

setup(
    name="mGPLVM",
    author="Ta-Chu Kao and Kris Jensen",
    version="0.0.1",
    description="Pytorch implementation of mGPLVM",
    license="MIT",
    install_requires=["numpy", "torch==1.7", "scipy>=1.0.0", "scikit-learn"],
    packages=find_packages(),
)

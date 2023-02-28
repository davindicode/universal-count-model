import os

from setuptools import find_packages, setup

setup(
    name="neuroprob",
    author="David Liu",
    version="1.0",
    description="PyTorch implementation of probabilistic neural encoding models",
    license="MIT",
    install_requires=["numpy", "scipy", "torch>=1.8", "tqdm", "matplotlib>=3.5", "daft"],
    packages=find_packages(),
)

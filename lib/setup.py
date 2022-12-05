import os

from setuptools import find_packages, setup

setup(
    name="neuroprob",
    author="David Liu",
    version="0.1",
    description="Pytorch implementation of UCM and probabilistic neural encoding models",
    license="MIT",
    install_requires=["numpy", "torch>=1.8", "scipy>=1.0.0", "daft", "tqdm"],
    packages=find_packages(),
)

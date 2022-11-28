# Neural data analysis library

Here we present a short description of the neural data analysis library used (called "neuroprob") to facilitate constructing scalable probabilsitic neural encoding models.
The list below shows what has implemented for use so far, see the models.py file for an example of code utilizing the library.



### Setup

```sh
# inside virtual environment
pip install -e .



### Primitives

There are three kinds of objects that form the building blocks:
1. Input group *p(X,Z)* and *q(Z)*
2. Mapping *p(F|X,Z)*
3. Likelihood *p(Y|F)*

The overal generative model is specified along with the variational posterior through these primitives.
Input groups can contain observed and latent variables, with different priors one can put onto the latent variables.


### Models implemented

* Linear-nonlinear and GP mappings
* LVMs
    - Toroidal latent space priors ([Manifold GPLVM](https://arxiv.org/abs/2006.07429))
    - AR(1) temporal prior on latents
* Count process likelihoods
    - Poisson
    - Zero-inflated Poisson
    - Negative binomial
    - Conway-Maxwell-Poisson
    - Universal (this work)
* Gaussian likelihoods
    - Univariate

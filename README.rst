.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/susiepca.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/susiepca
    .. image:: https://readthedocs.org/projects/susiepca/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://susiepca.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/susiepca/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/susiepca
    .. image:: https://img.shields.io/pypi/v/susiepca.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/susiepca/
    .. image:: https://img.shields.io/conda/vn/conda-forge/susiepca.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/susiepca
    .. image:: https://pepy.tech/badge/susiepca/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/susiepca
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/susiepca

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

========
SuSiE-PCA
========


    SuSiE PCA is a scalable Bayesian variable selection technique for sparse principal component analysis


SuSiE PCA is the abbreviation for the sum of single effects model in principal component analysis (SuSiE PCA). We develop SuSiE PCA for an efficient variable selection in PCA when dealing with high dimensional data with sparsity, and for quantifying uncertainty of contributing features for each latent component through posterior inclusion probabilities (PIPs). We implement the model with the [JAX](#https://github.com/google/jax) library developed by Google which enable the fast training on CPU, GPU or TPU. 

Here we introduce how to install SuSiE PCA and show the example of implementing it on the simulated data set in python.


========
Quick start
========

Install SuSiE PCA
====
The source code for SuSiE PCA is written fully in python 3.8. Follow these steps to quickly get started using SuSiE PCA.

1.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

=========
SuSiE-PCA
=========

    SuSiE PCA is a scalable Bayesian variable selection technique for sparse principal component analysis


SuSiE PCA is the abbreviation for the sum of single effects model in principal component analysis (SuSiE PCA). We
develop SuSiE PCA for an efficient variable selection in PCA when dealing with high dimensional data with sparsity, and
for quantifying uncertainty of contributing features for each latent component through posterior inclusion probabilities
(PIPs). We implement the model with the `JAX <https://github.com/google/jax>`_ library developed by Google which enable
the fast training on CPU, GPU or TPU.


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Overview <readme>

.. toctree::
   :maxdepth: 2
   :caption: API

   SuSiE-PCA API Reference <susiepca>

.. toctree::
   :maxdepth: 2
   :caption: Development

   Contributions & Help <contributing>
   Changelog <changelog>
   Authors <authors>
   License <license>

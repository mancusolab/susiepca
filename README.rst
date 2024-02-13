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

.. _Documentation: https://mancusolab.github.io/susiepca/
.. |Documentation| replace:: **Documentation**

=========
SuSiE-PCA
=========

    SuSiE PCA is a scalable Bayesian variable selection technique for sparse principal component analysis


SuSiE PCA is the abbreviation for the Sum of Single Effects model [1]_ for principal component analysis. We develop SuSiE PCA
for an efficient variable selection in PCA when dealing with high dimensional data with sparsity, and for quantifying
uncertainty of contributing features for each latent component through posterior inclusion probabilities (PIPs). We
implement the model with the `JAX <https://github.com/google/jax>`_ library developed by Google which enable the fast
training on CPU, GPU or TPU. The paper has been published in `iScience (2023) <https://www.cell.com/iscience/fulltext/S2589-0042(23)02258-7?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2589004223022587%3Fshowall%3Dtrue>`_

|Documentation|_ | |Installation|_ | |Example|_ | |Notes|_ | |References|_ | |Support|_

Model Description
=================
We extend the Sum of Single Effects model (i.e. SuSiE) [1]_ to principal component analysis. Assume $X_{N \\times P}$
is the observed data, $Z_{N \\times K}$ is the latent factors, and $W_{K \\times P}$ is the factor loading matrix, then
the SuSiE PCA model is given by:

$$X | Z,W \\sim \\mathcal{MN}_{N,P}(ZW, I_N, \\sigma^2 I_P)$$

where the $\\mathcal{MN}_{N,P}$ is the matrix normal distribution with dimension $N \\times P$,
mean $ZW$, row-covariance $I_N$, and column-covariance $I_P$. The column vector of $Z$ follows a
standard normal distribution. The above model setting is the same as the Probabilistic PCA [2]_. The
most distinguished part is that we integrate the SuSiE setting into the row vector $\\mathbf{w}_k$ of
factor loading matrix $W$, such that each $\\mathbf{w}_k$ only contains at most $L$ number of non-zero effects. That is,
$$\\mathbf{w}_k = \\sum_{l=1}^L \\mathbf{w}_{kl} $$
$$\\mathbf{w}_{kl} = w_{kl} \\gamma_{kl}$$
$$w_{kl} \\sim \\mathcal{N}(0,\\sigma^2_{0kl})$$
$$\\gamma_{kl} | \\pi \\sim \\text{Multi}(1,\\pi) $$

Notice that each row vector $\\mathbf{w}_k$ is a sum of single effect vector $\\mathbf{w}_{kl}$, which is length $P$ vector
contains only one non-zero effect $w_{kl}$ and zero elsewhere. And the coordinate of the non-zero effect is determined by
$\\gamma_{kl}$ that follows a multinomial distribution with parameter $\\pi$. By construction, each factor inferred from the
SuSiE PCA will have at most $L$ number of associated features from the original data. Moreover, we can quantify the probability
of the strength of association through the posterior inclusion probabilities (PIPs). Suppose the posterior distribution of
$\\gamma_{kl} \\sim \\text{Multi}(1,\\mathbf{\\alpha}_{kl})$, then the probability the feature $i$ contributing to the factor
$\\mathbf{w}_k$ is given by:
$$\\text{PIP}_{ki} = 1-\\prod_{l=1}^L (1 - \\alpha_{kli})$$
where the $\\alpha_{kli}$ is the $i_{th}$ entry of the $\\mathbf{\\alpha}_{kl}$.

.. _Installation:
.. |Installation| replace:: **Installation**

Install SuSiE PCA
=================
The source code for SuSiE PCA is written fully in Python 3.8 with JAX (see
`JAX installation guide <https://github.com/google/jax#installation>`_ for JAX). Follow the code provided below to quickly
get started using SuSiE PCA. Users can clone this github repository and install the SuSiE PCA. (Pypi installation will
be supported soon).

.. code:: bash

   git clone https://github.com/mancusolab/susiepca.git
   cd susiepca
   pip install -e .

.. _Example:
.. |Example| replace:: **Example**

Get Started with Example
========================

1. Create a python environment in the cloned repository, then simply import the SuSiE PCA

.. code:: python

   import susiepca as sp

2. Generate a simulation data set according to the description in **Simulation** section from our paper. $Z_{N \\times K}$
   is the simulated factors matrix, $W_{K \\times P}$ is the simulated loading matrix, and the $X_{N \\times P}$ is the
   simulation data set that has $N$ observations with $P$ features.

.. code:: python

   Z, W, X = sp.sim.generate_sim(seed = 0, l_dim = 40, n_dim = 150, p_dim =200, z_dim = 4, effect_size = 1)

3. Input the simulation data set into SuSiE PCA with number of component $K=4$ and number of single effects in each component $L=40$, or you can manipulate with those two parameters to check the model mis-specification performance. By default the data is not centered nor scaled, and the max iteration is set to be 200. Here we use the principal components extracted from traditional PCA results as the initialization of mean of $Z$.

.. code:: python

   results = sp.infer.susie_pca(X, z_dim = 4, l_dim = 40, max_iter=200)

The returned "results" contain 5 different objects:

- params: an dictionary that saves all the updated parameters from the SuSiE PCA.
- elbo_res: the value of evidence lower bound (ELBO) from the last iteration.
- pve: a length $K$ ndarray contains the percent of variance explained (PVE) by each component.
- pip: the $K$ by $P$ ndarray that contains the posterior inclusion probabilities (PIPs) of each feature contribution to the factor.
- W: the posterior mean of loading matrix which is also a $K$ by $P$ ndarray.

4. To examine the model performance, one straitforward way is to draw and compare the heatmap of the true loading matrix
   and estimate loading matrix using seaborn:

.. code:: python

   import seaborn as sns

   # specify the palatte for heatmap
   div = sns.diverging_palette(250, 10, as_cmap=True)

   # Heatmap of true loading matrix
   sns.heatmap(W, cmap = div, fmt = ".2f",center = 0)

   # Heatmap of estimate loading matrix
   W_hat = results.W
   sns.heatmap(W_hat, cmap = div, fmt = ".2f", center = 0)

   # Heatmap of PIPs
   pip = results.pip
   sns.heatmap(pip, cmap = div, fmt = ".2f", center = 0)

To mathmatically compute the Procrustes error of the estimate loading matrix, you need to install the Procruste package
to solve the rotation problem (see `procrustes installation guide <https://procrustes.readthedocs.io/en/latest/usr_doc_installization.html>`_
for Procrustes method). Once the loading matrix is rotated to its original direction, one can compute the Procrustes error and look at heatmap as following:

.. code:: python

   import procrustes
   import numpy as np

   # perform procrustes transformation
   proc_trans_susie = procrustes.orthogonal(np.asarray(W_hat.T), np.asarray(W.T), scale=True)
   print(f"The Procrustes error for the loading matrix is {proc_trans_susie.error}")
   
   # Heatmap of transformed loading matrix
   W_trans = proc_trans_susie.t.T @ W_hat
   sns.heatmap(W_trans, cmap = div, fmt = ".2f", center = 0)

You can also calculate the relative root mean square error (RRMSE) to assess the model prediction performance

.. code:: python

   from susiepca import metrics

   # compute the predicted data
   X_hat = results.params.mu_z @ W_hat

   # compute the RRMSE
   rrmse_susie = metrics.mse(X, X_hat)

5. Finally we also provide a neat function to compute a $\\rho-$ level credible sets (CS). The cs returned by the function is composed of $L \\times K$ credible sets, each of them contain a subset of variables that cumulatively explain at least $\\rho$ of the posterior density.

.. code:: python

   cs = sp.metrics.get_credset(results.params.alpha, rho=0.9)

.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====

`JAX <https://github.com/google/jax>`_ uses 32-bit precision by default. To enable 64-bit precision before calling
`susiepca` add the following code:

.. code:: python

   import jax
   jax.config.update("jax_enable_x64", True)

Similarly, the default computation device for `JAX <https://github.com/google/jax>`_ is set by environment variables
(see `here <https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement>`_). To change this programmatically before
calling `susiepca` add the following code:

.. code:: python

   import jax
   platform = "gpu" # "gpu", "cpu", or "tpu"
   jax.config.update("jax_platform_name", platform)

.. _References:
.. |References| replace:: **References**

References
==========
.. [1] Wang, G., Sarkar, A., Carbonetto, P. and Stephens, M. (2020), A simple new approach to variable selection in regression, with application to genetic fine mapping. J. R. Stat. Soc. B, 82: 1273-1300. https://doi.org/10.1111/rssb.12388
.. [2] Tipping, M.E. and Bishop, C.M. (1999), Probabilistic Principal Component Analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61: 611-622. https://doi.org/10.1111/1467-9868.00196

.. _Support:
.. |Support| replace:: **Support**

Support
=======
Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/susiepca/issues>`_. If you have any 
questions or comments please contact dongyuan@usc.edu and/or nmancuso@usc.edu. 

---------------------

.. _pyscaffold-notes:

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

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


SuSiE PCA is the abbreviation for the sum of single effects model in principal component analysis (SuSiE PCA). We develop SuSiE PCA for an efficient variable selection in PCA when dealing with high dimensional data with sparsity, and for quantifying uncertainty of contributing features for each latent component through posterior inclusion probabilities (PIPs). We implement the model with the `JAX <https://github.com/google/jax>`_ library developed by Google which enable the fast training on CPU, GPU or TPU. 

Next we introduce how to install SuSiE PCA and show the example of implementing it on the simulated data set in python.

========
Quick start
========

Install SuSiE PCA
====
The source code for SuSiE PCA is written fully in python 3.8. Follow these steps to quickly get started using SuSiE PCA.

1. Install the JAX through the following code. Here we provide the installation of CPU-only version of JAX. For the GPU version, check the JAX installation guide (https://github.com/google/jax).

::

   pip install --upgrade pip
   pip install --upgrade "jax[cpu]"

2. Then you can clone this github repository in the desired directorty through:

::

   git clone git@github.com:mancusolab/susiepca.git
   

Get Started with Example
====

1. Create a python environment in the cloned repository, then simply import the SuSiE PCA

::

   import susiepca as sp

2. Generate a simulation data set according to the description in Simulation section from our paper. $Z_{N \\times K}$ is the simulated factors matrix, $W_{K \\times P}$ is the simulated loading matrix, and the $X_{N \\times P}$ is the simulation data set that has $N$ observations with $P$ features.

::

   Z, W, X = sp.sim.generate_sim(seed = 0, l_dim = 40,n_dim = 150, p_dim =200, z_dim = 4, effect_size = 1)

3. Input the simulation data set into SuSiE PCA with $K=4$ and $L=40$, or you can manipulate with those two parameters to check the model mis-specification performance. 

::

   results = sp.infer.susie_pca(X, z_dim = 4, l_dim = 40, max_iter=200)

The returned "results" contain 5 different objects:

- params: an dictionary that saves all the updated parameters from the SuSiE PCA.
- elbo_res: the value of evidence lower bound (ELBO) from the last iteration.
- pve: a length $K$ ndarray contains the percent of variance explained (PVE) by each component
- pip: the ndarray in dimension of $K$ by $P$ that contains the posterior inclusion probabilities of each feature contribution to each factor.
- W: the posterior mean of loadings which is also a ndarray in dimension of $K$ by $P$ 

4. To examine the model performance, one straitforward way is to draw and compare the heatmap of the true loading matrix and estimate loading matrix using seaborn:

::

   import seaborn as sns
   div = sns.diverging_palette(250, 10, as_cmap=True)
   #Heatmap of true loading matrix
   sns.heatmap(W, cmap = div,fmt = ".2f",center = 0)
   #Heatmap of estimate loading matrix
   W_hat = results.W
   sns.heatmap(W_hat, cmap = div,fmt = ".2f",center = 0)

To mathmatically compute the Procrustes error of the estimate loading matrix, you need to install the `Procruste <https://procrustes.readthedocs.io/en/latest/usr_doc_installization.html>`_ package.

::
   
   pip install qc-procrustes
   import procrutes
   #peform procruste transformation
   proc_trans_susie = procrustes.orthogonal(np.asarray(W_hat.T),np.asarray(W.T),scale=True)
   print(f"The Procrustes error for the loading matrix is {proc_trans_susie.error}")

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.



======
SuSiE-PCA
======

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
    Quickstart <quickstart>
    Module Reference <api>

.. toctree::
    :caption: Development

    Contributions & Help <contributing>
    Changelog <changelog>
    Authors <authors>
    License <license>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists

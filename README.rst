=====================================
fgivenx: Functional Posterior Plotter  
=====================================
:fgivenx:  Functional Posterior Plotter 
:Author: Will Handley
:Version: 2.1.9
:Homepage: https://github.com/williamjameshandley/fgivenx
:Documentation: http://fgivenx.readthedocs.io/

.. image:: https://travis-ci.org/williamjameshandley/fgivenx.svg?branch=master
   :target: https://travis-ci.org/williamjameshandley/fgivenx
   :alt: Build Status
.. image:: https://codecov.io/gh/williamjameshandley/fgivenx/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/williamjameshandley/fgivenx
   :alt: Test Coverage Status
.. image:: https://badge.fury.io/py/fgivenx.svg
   :target: https://badge.fury.io/py/fgivenx
   :alt: PyPi location
.. image:: https://readthedocs.org/projects/fgivenx/badge/?version=latest
   :target: https://fgivenx.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: http://joss.theoj.org/papers/cf6f8ac309d6a18b6d6cf08b64aa3f62/status.svg
   :target: http://joss.theoj.org/papers/cf6f8ac309d6a18b6d6cf08b64aa3f62
   :alt: Review Status
.. image:: https://zenodo.org/badge/100947684.svg
   :target: https://zenodo.org/badge/latestdoi/100947684
   :alt: Permanent DOI

Description
===========

``fgivenx`` is a python package for plotting posteriors of functions. It is
currently used in astronomy, but will be of use to any scientists performing
Bayesian analyses which have predictive posteriors that are functions.

These packages allow one to plot a predictive posterior of a function,
dependent on sampled parameters. We assume one has a Bayesian posterior
``Post(theta|D,M)`` described by a set of posterior samples ``{theta_i}~Post``.
If there is a function parameterised by theta ``y=f(x;theta)``, then this script
will produce a contour plot of the conditional posterior ``P(y|x,D,M)`` in the
``(x,y)`` plane.

The driving routines are ``fgivenx.plot_contours``, ``fgivenx.plot_lines`` and
``fgivenx.plot_dkl``. The code is compatible with getdist, and has a loading function
provided by ``fgivenx.samples_from_getdist_chains``.

|image0|

Getting Started
===============

Users can install using pip:

.. code:: bash

   pip install fgivenx

from source:

.. code:: bash

   git clone https://github.com/williamjameshandley/fgivenx
   cd fgivenx
   python setup.py install --user

or for those on `Arch linux <https://www.archlinux.org/>`__ it is
available on the
`AUR <https://aur.archlinux.org/packages/python-fgivenx/>`__

You can check that things are working by running the test suite:

.. code:: bash

   export MPLBACKEND=Agg
   pytest
   # or, equivalently
   python setup.py test

Check the dependencies listed in the next section are installed. You can then
use the ``fgivenx`` module from your scripts.

If you want to use parallelisation, have progress bars or getdist compatibility
you should install the additional optional dependencies:

.. code:: bash

   pip install joblib tqdm getdist
   # or, equivalently
   pip install -r  requirements.txt

Dependencies
=============
Basic requirements:

* Python 2.7+ or 3.4+
* `matplotlib <https://pypi.org/project/matplotlib/>`__
* `numpy <https://pypi.org/project/numpy/>`__
* `scipy <https://pypi.org/project/scipy/>`__

Documentation:

* `sphinx <https://pypi.org/project/Sphinx/>`__
* `numpydoc <https://pypi.org/project/numpydoc/>`__

Tests:

* `pytest <https://pypi.org/project/pytest/>`__

Optional extras:

* `joblib <https://pypi.org/project/joblib/>`__ (parallelisation)
* `tqdm <https://pypi.org/project/tqdm/>`__ (progress bars)
* `getdist <https://pypi.org/project/GetDist/>`__ (reading of getdist compatible files)


Documentation
=============

Full Documentation is hosted at
`ReadTheDocs <http://fgivenx.readthedocs.io/>`__.
To build your own local copy of the documentation you'll need to install
`sphinx <https://pypi.org/project/Sphinx/>`__. You can then run:

.. code:: bash

   cd docs
   make html

Citation
========

If you use ``fgivenx`` to generate plots for a publication, please cite
as:

.. code:: bibtex

   @article{fgivenx,
       doi = {10.21105/joss.00849},
       url = {http://dx.doi.org/10.21105/joss.00849},
       year  = {xxxx},
       month = {xxx},
       publisher = {The Open Journal},
       volume = {X},
       number = {X},
       author = {Will Handley},
       title = {fgivenx: Functional Posterior Plotter},
       journal = {The Journal of Open Source Software}
   }

Example Usage
=============



Plot user-generated samples
---------------------------

.. code:: python

    import numpy
    import matplotlib.pyplot as plt
    from fgivenx import plot_contours, plot_lines, plot_dkl


    # Model definitions
    # =================
    # Define a simple straight line function, parameters theta=(m,c)
    def f(x, theta):
        m, c = theta
        return m * x + c


    numpy.random.seed(1)

    # Posterior samples
    nsamples = 1000
    ms = numpy.random.normal(loc=-5, scale=1, size=nsamples)
    cs = numpy.random.normal(loc=2, scale=1, size=nsamples)
    samples = numpy.array([(m, c) for m, c in zip(ms, cs)]).copy()

    # Prior samples
    ms = numpy.random.normal(loc=0, scale=5, size=nsamples)
    cs = numpy.random.normal(loc=0, scale=5, size=nsamples)
    prior_samples = numpy.array([(m, c) for m, c in zip(ms, cs)]).copy()

    # Set the x range to plot on
    xmin, xmax = -2, 2
    nx = 100
    x = numpy.linspace(xmin, xmax, nx)

    # Set the cache
    cache = 'cache/test'
    prior_cache = cache + '_prior'

    # Plotting
    # ========
    fig, axes = plt.subplots(2, 2)

    # Sample plot
    # -----------
    ax_samples = axes[0, 0]
    ax_samples.set_ylabel(r'$c$')
    ax_samples.set_xlabel(r'$m$')
    ax_samples.plot(prior_samples.T[0], prior_samples.T[1], 'b.')
    ax_samples.plot(samples.T[0], samples.T[1], 'r.')

    # Line plot
    # ---------
    ax_lines = axes[0, 1]
    ax_lines.set_ylabel(r'$y = m x + c$')
    ax_lines.set_xlabel(r'$x$')
    plot_lines(f, x, prior_samples, ax_lines, color='b', cache=prior_cache)
    plot_lines(f, x, samples, ax_lines, color='r', cache=cache)

    # Predictive posterior plot
    # -------------------------
    ax_fgivenx = axes[1, 1]
    ax_fgivenx.set_ylabel(r'$P(y|x)$')
    ax_fgivenx.set_xlabel(r'$x$')
    cbar = plot_contours(f, x, prior_samples, ax_fgivenx,
                         colors=plt.cm.Blues_r, lines=False,
                         cache=prior_cache)
    cbar = plot_contours(f, x, samples, ax_fgivenx, cache=cache)

    # DKL plot
    # --------
    ax_dkl = axes[1, 0]
    ax_dkl.set_ylabel(r'$D_\mathrm{KL}$')
    ax_dkl.set_xlabel(r'$x$')
    ax_dkl.set_ylim(bottom=0)
    plot_dkl(f, x, samples, prior_samples, ax_dkl,
             cache=cache, prior_cache=prior_cache)

    ax_lines.get_shared_x_axes().join(ax_lines, ax_fgivenx, ax_samples)

    fig.tight_layout()
    fig.savefig('plot.png')

|image0|

Plot GetDist chains
-------------------

.. code:: python

    import numpy
    import matplotlib.pyplot as plt
    from fgivenx import plot_contours, samples_from_getdist_chains

    file_root = './plik_HM_TT_lowl/base_plikHM_TT_lowl'
    samples, weights = samples_from_getdist_chains(['logA', 'ns'], file_root)

    def PPS(k, theta):
        logA, ns = theta
        return logA + (ns - 1) * numpy.log(k)
        
    k = numpy.logspace(-4,1,100)
    cbar = plot_contours(PPS, k, samples, weights=weights)
    cbar = plt.colorbar(cbar,ticks=[0,1,2,3])
    cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
    
    plt.xscale('log')
    plt.ylim(2,4)
    plt.ylabel(r'$\ln\left(10^{10}\mathcal{P}_\mathcal{R}\right)$')
    plt.xlabel(r'$k / {\rm Mpc}^{-1}$')
    plt.tight_layout()
    plt.savefig('planck.png')

|image1|

Contributing
============
Want to contribute to ``fgivenx``? Awesome!
There are many ways you can contribute via the 
[GitHub repository](https://github.com/williamjameshandley/fgivenx), 
see below.

Opening issues
--------------
Open an issue to report bugs or to propose new features.

Proposing pull requests
-----------------------
Pull requests are very welcome. Note that if you are going to propose drastic
changes, be sure to open an issue for discussion first, to make sure that your
PR will be accepted before you spend effort coding it.

.. |image0| image:: https://github.com/williamjameshandley/fgivenx/blob/master/plot.png
.. |image1| image:: https://github.com/williamjameshandley/fgivenx/blob/master/planck.png


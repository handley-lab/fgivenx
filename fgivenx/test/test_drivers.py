import pytest
import numpy
import matplotlib.pyplot as plt
from fgivenx import plot_contours, plot_lines, plot_dkl
from fgivenx.drivers import compute_samples, compute_pmf, compute_dkl
from matplotlib.colors import LogNorm


def test_full():
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

    # Computation
    # ===========
    # Examine the function over a range of x's
    xmin, xmax = -2, 2
    nx = 100
    x = numpy.linspace(xmin, xmax, nx)

    # Set the cache
    cache = 'cache/test'
    prior_cache = cache + '_prior'

    # Compute function samples
    with pytest.raises(TypeError):
        compute_samples(f, x, samples, unknown_arg=None)
    compute_samples(f, x, samples, cache=cache)
    compute_samples(f, x, prior_samples, cache=prior_cache)

    # Compute dkls
    with pytest.raises(TypeError):
        compute_dkl(f, x, samples, prior_samples, unknown_arg=None)
    compute_dkl(f, x, samples, prior_samples,
                cache=cache, parallel=True)

    # Compute probability mass function.
    with pytest.raises(TypeError):
        compute_pmf(f, x, samples, unknown_arg=None)
    with pytest.raises(ValueError):
        compute_pmf(f, x, samples, y=numpy.random.rand(2, 2))
    y, pmf = compute_pmf(f, x, samples, cache=cache, parallel=True)
    y_prior, pmf_prior = compute_pmf(f, x, prior_samples,
                                     cache=prior_cache, parallel=True)

    plot_dkl(f, x, samples, prior_samples,
             cache=cache, prior_cache=prior_cache)


def test_plotting():
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

    # Examine the function over a range of x's
    xmin, xmax = -2, 2
    nx = 100
    x = numpy.linspace(xmin, xmax, nx)

    # Set the cache
    for cache in [None, 'cache/test']:
        if cache is not None:
            prior_cache = cache + '_prior'
        else:
            prior_cache = None

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
        plot_contours(f, x, prior_samples, ax_fgivenx,
                      colors=plt.cm.Blues_r, lines=False,
                      cache=prior_cache)
        plot_contours(f, x, samples, ax_fgivenx, cache=cache)

        # DKL plot
        # --------
        ax_dkl = axes[1, 0]
        ax_dkl.set_ylabel(r'$D_\mathrm{KL}$')
        ax_dkl.set_xlabel(r'$x$')
        ax_dkl.set_ylim(bottom=0)
        plot_dkl(f, x, samples, prior_samples, ax_dkl,
                 cache=cache, prior_cache=prior_cache)

        ax_lines.get_shared_x_axes().join(ax_lines, ax_fgivenx, ax_samples)
        fig.set_size_inches(6, 6)

def test_histogram():
    # Model definitions
    # =================
    # Define a simple straight line function, parameters theta=(m,c)
    def f(x, theta):
        m, c = theta
        return m * x + c

    numpy.random.seed(1)

    # Posterior samples
    nsamples = 100000
    ms = numpy.random.normal(loc=-5, scale=1, size=nsamples)
    cs = numpy.random.normal(loc=2, scale=1, size=nsamples)
    samples = numpy.array([(m, c) for m, c in zip(ms, cs)]).copy()

    # Prior samples
    ms = numpy.random.normal(loc=0, scale=5, size=nsamples)
    cs = numpy.random.normal(loc=0, scale=5, size=nsamples)
    prior_samples = numpy.array([(m, c) for m, c in zip(ms, cs)]).copy()

    # Examine the function over a range of x's
    xmin, xmax = -2, 2
    nx = 1000
    nx_kde = 100
    ny = 200
    ny_kde = 100
    x = numpy.linspace(xmin, xmax, nx)
    x_kde = numpy.linspace(xmin, xmax, nx_kde)
    samples_kde = samples[::10]

    # Demonstrate nice cmap
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    level_cmap = LinearSegmentedColormap.from_list('level_cmap',
                                                   ["red", "orange", "white"],
                                                   N=5)

    # Set the cache
    for cache in [None, 'cache/test']:
        # Plotting
        # ========
        fig, axes = plt.subplots()
        ax_fgivenx = axes
        ax_fgivenx.set_ylabel(r'$P(y|x)$')
        ax_fgivenx.set_xlabel(r'$x$')
        cbar = plot_contours(f, x, samples, ax_fgivenx, cache=cache,
                             histogram=True, ny=ny, colors=level_cmap,
                             fineness=1, contour_line_levels=[1,2,3,4,5])
        plot_contours(f, x_kde, samples_kde, ax_fgivenx, cache=cache,
                      alpha=0, linewidths=1, ny=ny_kde,
                      contour_line_levels=[1,2,3,4,5])
        fig.colorbar(cbar, label=r"$\sigma$")

        fig, axes = plt.subplots()
        ax_fgivenx = axes
        ax_fgivenx.set_ylabel(r'$P(y|x)$')
        ax_fgivenx.set_xlabel(r'$x$')
        cbar = plot_contours(f, x, samples, ax_fgivenx, cache=cache,
                             histogram=True, ny=ny, pdf_histogram=True)
        plot_contours(f, x_kde, samples_kde, ax_fgivenx, cache=cache,
                      alpha=0, linewidths=1, ny=ny_kde,
                      contour_line_levels=[1,2,3,4,5])
        fig.colorbar(cbar, label=r"PDF")

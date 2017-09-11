Functional Posterior Plotter
============================

These packages allow one to compute a predictive posterior of a function,
dependent on sampled parameters. We assume one has a Bayesian posterior
Post(theta|D,M) described by a set of posterior samples {theta_i}~Post. If
there is a function parameterised by theta f(x;theta), then this script will
produce a contour plot of the conditional posterior P(f|x,D,M) in the (x,f)
plane.

The driving routine is `fgivenx.compute_contours`, and example usage can be
found by running `help(fgivenx)`. It is compatible with getdist, and has a
loading function provided by `fgivenx.samples.samples_from_getdist_chains`.

Example Usage
-------------

```python
import numpy
import matplotlib.pyplot
from fgivenx import compute_samples, compute_contours, compute_kullback_liebler
import fgivenx.plot


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
x, fsamps = compute_samples(f, x, samples, cache=cache)
x, prior_fsamps = compute_samples(f, x, prior_samples, cache=prior_cache)

# Compute dkls
x, dkls = compute_kullback_liebler(f, x, samples, prior_samples, cache=cache)

# Compute probability mass function.
x, y, z = compute_contours(f, x, samples, cache=cache)
x, y_prior, z_prior = compute_contours(f, x, prior_samples, cache=prior_cache)

# Plotting
# ========
fig, axes = matplotlib.pyplot.subplots(2, 2)
prior_color = 'b'
posterior_color = 'r'

# Sample plot
# -----------
ax = axes[0, 0]
ax.set_ylabel(r'$c$')
ax.set_xlabel(r'$m$')
ax.plot(prior_samples.T[0], prior_samples.T[1],
        color=prior_color, linestyle='.')
ax.plot(samples.T[0], samples.T[1],
        color=posterior_color, linestyle='.')

# Line plot
# ---------
ax = axes[0, 1]
ax.set_ylabel(r'$y = m x + c$')
fgivenx.plot.plot_lines(x, prior_fsamps, ax, color=prior_color)
fgivenx.plot.plot_lines(x, fsamps, ax, color=posterior_color)
ax.set_xticklabels([])

# Predictive posterior plot
# -------------------------
ax = axes[1, 1]
ax.set_ylabel(r'$P(y|x)$')
ax.set_xlabel(r'$x$')
cbar = fgivenx.plot.plot(x, y_prior, z_prior, ax,
                         colors=matplotlib.pyplot.cm.Blues_r,
                         lines=False)
cbar = fgivenx.plot.plot(x, y, z, ax,
                         colors=matplotlib.pyplot.cm.Reds_r)

# DKL plot
# --------
ax = axes[1, 0]
ax.set_ylabel(r'$D_\mathrm{KL}$')
ax.set_xlabel(r'$x$')
ax.plot(x, dkls)
ax.set_ylim(bottom=0)

axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0], axes[1, 1])

fig.tight_layout()
fig.savefig('plot.png')
```
![](https://raw.github.com/williamjameshandley/fgivenx/master/plot.png)





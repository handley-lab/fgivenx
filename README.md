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
import matplotlib.pyplot as plt
from fgivenx import compute_samples, compute_pmf, compute_dkl
from fgivenx.plot import plot, plot_lines


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
fsamps = compute_samples(f, x, samples, cache=cache)
prior_fsamps = compute_samples(f, x, prior_samples, cache=prior_cache)

# Compute dkls
dkls = compute_dkl(f, x, samples, prior_samples, cache=cache, parallel=True)

# Compute probability mass function.
y, pmf = compute_pmf(f, x, samples, cache=cache, parallel=True)
y_prior, pmf_prior = compute_pmf(f, x, prior_samples, cache=prior_cache, parallel=True)

# Plotting
# ========
fig, axes = plt.subplots(2, 2)
prior_color = 'b'
posterior_color = 'r'

# Sample plot
# -----------
ax_samples = axes[0, 0]
ax_samples.set_ylabel(r'$c$')
ax_samples.set_xlabel(r'$m$')
ax_samples.plot(prior_samples.T[0], prior_samples.T[1], color=prior_color, marker='.', linestyle='')
ax_samples.plot(samples.T[0], samples.T[1], color=posterior_color, marker='.', linestyle='')

# Line plot
# ---------
ax_lines = axes[0, 1]
ax_lines.set_ylabel(r'$y = m x + c$')
ax_lines.set_xlabel(r'$x$')
plot_lines(x, prior_fsamps, ax_lines, color=prior_color)
plot_lines(x, fsamps, ax_lines, color=posterior_color)

# Predictive posterior plot
# -------------------------
ax_fgivenx = axes[1, 1]
ax_fgivenx.set_ylabel(r'$P(y|x)$')
ax_fgivenx.set_xlabel(r'$x$')
cbar = plot(x, y_prior, pmf_prior, ax_fgivenx, colors=plt.cm.Blues_r, lines=False)
cbar = plot(x, y, pmf, ax_fgivenx, colors=plt.cm.Reds_r)

# DKL plot
# --------
ax_dkl = axes[1, 0]
ax_dkl.set_ylabel(r'$D_\mathrm{KL}$')
ax_dkl.set_xlabel(r'$x$')
ax_dkl.plot(x, dkls)
ax_dkl.set_ylim(bottom=0)

ax_lines.get_shared_x_axes().join(ax_lines, ax_fgivenx, ax_samples)

fig.tight_layout()
fig.savefig('plot.pdf')
```
![](https://raw.github.com/williamjameshandley/fgivenx/master/plot.pdf)





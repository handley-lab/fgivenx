"""

The main driving routines for this package are:

* :func:`plot_contours <fgivenx.drivers.plot_contours>`
* :func:`plot_lines <fgivenx.drivers.plot_lines>`
* :func:`plot_dkl <fgivenx.drivers.plot_dkl>`
* :func:`samples_from_getdist_chains <fgivenx.samples.samples_from_getdist_chains>`

Example import and usage:

>>> import numpy
>>> from fgivenx import plot_contours, plot_lines, plot_dkl, samples_from_getdist_chains
>>> 
>>> file_root = '/my/getdist/file/root'
>>> params = ['m', 'c']
>>> samples = samples_from_getdist_chains(params, file_root)
>>> x = numpy.linspace(-1, 1, 100)
>>> 
>>> def f(x, theta):
>>>     m, c = params
>>>     y = m * x + c
>>>     return y
>>> 
>>> plot_contours(f, x, samples)
"""

from fgivenx.drivers import plot_contours, plot_lines, plot_dkl
from fgivenx.samples import samples_from_getdist_chains

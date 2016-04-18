#!/usr/bin/python

# This script computes the grid for contours of a function
# reconstruction plot.
#
# If one has
#  * independent variable x
#  * dependent variable y
#  * functional form y = f(x,theta) parameterised by theta
#
# This script assumes a linear spline, with theta being interpolation
# knots, but the file fgivenx/samples.py can easily be modified
#
# Assuming that you have obtained samples of theta from an MCMC
# process, we aim to compute:
#
#               /
# P( y | x ) =  | P( y = f(x,theta) | x, theta ) dtheta ,  (1)
#               /
#
# which gives our degree of knowledge for each y value given an x value.
#
# In fact, for a more representative plot, we are not actually
# interested in the value of the probability density (1), but in fact
# require the "iso-probablity posterior mass:"
#
#                     /
# m( y | x ) =        | P(y'|x) dy'
#                     /
#             P(y'|x) < P(y|x)
#
# We thus need to compute this function on a rectangular grid of x and y's
#
# Once this is done, you should then use the plot_fgivenx.py script
#
# If you encounter an error, the first thing to check should be the
# limits: xmin,xmax,ymin,ymax
#
# Any questions, please email Will Handley <wh260@mrao.cam.ac.uk>

import numpy
from fgivenx.contours import compute_contour_plot
from fgivenx.read_data import save_contours
from fgivenx.sample import trim_samples

from fgivenx.data_storage import FunctionalPosterior
import sys

import numpy

# Load posteriors 
chains_file = 'chains/mps10_detec_nliv200_ident_sub.txt'
paramnames_file = 'chains/mps10_detec_nliv200_ident_sub.paramnames'

# Define the function
def logspectrum(logE, params):
    """ General spectrum """
    logN0, alpha, beta, logE0, logEc = params

    logdNdE = logN0
    logdNdE -= (logE - logE0) * (alpha + beta*(logE - logE0) )
    logdNdE -= numpy.exp(logE-logEc) - numpy.exp(logE0 - logEc)

    return logdNdE


def logspectrum_fix_E0_Ec(logE, params):
    """ Spectrum with logE0 and logEc fixed """
    logE0 = numpy.log(4016.0)
    logEc = float('inf')
    return logspectrum(logE, params + [logE0, logEc])

# Create a set of function posteriors
function = logspectrum_fix_E0_Ec
chosen_parameters = ['log_N0_1','alpha_PS_1','beta_PS_1']

posterior = FunctionalPosterior(chains_file,paramnames_file)
posterior.set_function(function,chosen_parameters)


# number of samples to keep ( <= 0 means keep all)
# Plots are quick to compute if nsamp~1000, and typically entirely stable
# if setting a low value of nsamp, users are recommend to run
# several plots and compare stability
nsamp = 100
posterior.trim_samples(nsamp)

print "Computing Contours"
print "------------------"

# Settings
# --------
nx   = 100                     # resolution in x direction (this is normally sufficient)
xmin = 2                       # minimum of x range
xmax = 12                      # maximum of x range

ny   = 100      # resolution in y direction (this is normally sufficient)
ymin = -20      # minimum of y range
ymax = -5       # maximum of y range


root = 'my_data'            # the root name for the other files

progress_bar = True

# Computing contours
# ------------------

# Read chains file and convert into a set of interpolation functions
#   - We assume that the chains file has the format:
#      <weight>    <N x coordinates>   <N y coordinates>
#     in a space - separated file (just as in getdist)
#
#   - Note that you may need to use python or awk to process a raw chains file
#
#   - It is fine for each line to have different values of N, but there
#     shouldn't be any additional irrelevant parameters
#

# Compute a grid for making a contour plot
x = numpy.linspace(xmin, xmax, nx)
y = numpy.linspace(ymin, ymax, ny)
z = compute_contour_plot(posterior, x, y, progress_bar)

# Save the contours to file for later use
save_contours(root, x, y, z)
print ""

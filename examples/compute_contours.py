#!/usr/bin/python
import numpy
from fgivenx.contours import Contours
from fgivenx.data_storage import FunctionalPosterior

# setup
# -----
xmin = 4                       # minimum of x range
xmax = 12                      # maximum of x range
nsamp = 500                    # number of samples to use
chains_file = 'chains/mps10_detec_nliv200_ident_sub.txt'
paramnames_file = 'chains/mps10_detec_nliv200_ident_sub.paramnames'

def f(logE, params):
    """ Spectrum with logE0 and logEc fixed """
    logE0 = numpy.log(4016.0)
    logEc = float('inf')
    logN0, alpha, beta = params

    logdNdE = logN0
    logdNdE -= (logE - logE0) * (alpha + beta*(logE - logE0) )
    logdNdE -= numpy.exp(logE-logEc) - numpy.exp(logE0 - logEc)

    return logdNdE

chosen_parameters = ['log_N0_1', 'alpha_1', 'beta_1']


# Computing contours
# ------------------

# load the posteriors from file
posterior = FunctionalPosterior(chains_file,paramnames_file).trim_samples(nsamp)

# Generate some functional posteriors
posterior.set_function(f, chosen_parameters)

# Compute the contours
contours = Contours(posterior,[xmin, xmax])

# Save the contours
contours.save('contours/posterior.pkl')

#!/usr/bin/python
import numpy
from fgivenx.contours import Contours
from fgivenx.data_storage import FunctionalPosterior

# setup
# -----
xmin = -5                                    # minimum of x range
xmax = 5                                     # maximum of x range
nsamp = 500                                  # number of samples to use
chains_file = 'chains/test.txt'              # posterior files
paramnames_file = 'chains/test.paramnames'   # paramnames  file

def f(x, theta):
    """ Simple y = m x + c """
    m, c = theta

    return m * x + c

chosen_parameters = ['m_1', 'c_1']


# Computing contours
# ------------------
# load the posteriors from file, and set the function
posterior = FunctionalPosterior(chains_file, paramnames_file).trim_samples(nsamp).set_function(f, chosen_parameters)

# Compute the contours and save
contours = Contours(posterior,[xmin, xmax])
contours.save('contours/posterior.pkl')

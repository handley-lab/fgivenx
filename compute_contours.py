#!/usr/bin/python

""" This script computes the grid for contours of a function
    reconstruction plot.
    
    If one has
     * independent variable x
     * dependent variable y
     * functional form y = f(x,theta) parameterised by theta
    
    Assuming that you have obtained samples of theta from an MCMC
    process, we aim to compute:
    
                  /
    P( y | x ) =  | P( y = f(x,theta) | x, theta ) dtheta ,  (1)
                  /
    
    which gives our degree of knowledge for each y value given an x value.
    
    In fact, for a more representative plot, we are not actually
    interested in the value of the probability density (1), but in fact
    require the "iso-probablity posterior mass:"
    
                        /
    m( y | x ) =        | P(y'|x) dy'
                        /
                P(y'|x) < P(y|x)
    
    We thus need to compute this function on a rectangular grid of x and y's
    
    Once this is done, you should then use the plot_fgivenx.py script
    
    Any questions, please email Will Handley <wh260@mrao.cam.ac.uk>
"""

import numpy
from fgivenx.contours import Contours
from fgivenx.data_storage import FunctionalPosterior

# Settings
# --------
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

choices = [['log_N0_' + str(i), 'alpha_PS_' + str(i), 'beta_PS_' + str(i)] for i in range(1,11)]
#interp1d(xdat,ydat,bounds_error=False,fill_value=0)


# load the posteriors from file
posterior = FunctionalPosterior(chains_file,paramnames_file).trim_samples(nsamp)

# Compute a grid for making a contour plot
for i, chosen_parameters in enumerate(choices):
    # Generate some functional posteriors
    posterior.set_function(f, chosen_parameters)

    # Compute the contours
    contours = Contours(posterior,[xmin, xmax],progress_bar=True)

    # Save the contours
    contours.save('contours/posterior' + str(i) + '.pkl')




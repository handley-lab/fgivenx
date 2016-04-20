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
    
    Any questions, please email Will Handley <wh260@mrao.cam.ac.uk>
"""

import pickle
import numpy
import matplotlib.pyplot

from fgivenx.utils import PMF
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
from tqdm import tqdm as pbar

def load_contours(datafile):
    return pickle.load(open(datafile,'r'))

class Contours(object):

    def __init__(self, fsamples, x_range, nx=200, ny='nx'):

        if ny == 'nx':
            ny = nx

        # Set up x coordinates
        self.x = numpy.linspace(x_range[0], x_range[1], nx)

        # Compute masses at each value of x
        masses = [PMF(fsamples(x)) for x in pbar(self.x, desc="computing masses")]

        # Compute upper and lower bounds on y
        self.upper = max([m.upper for m in masses])
        self.lower = min([m.lower for m in masses])

        # Set up y coordinates
        self.y = numpy.linspace(self.lower, self.upper, ny)
        
        # Compute densities across the grid
        self.z = [[m(y) for m in masses] for y in self.y]


    def save(self,datafile):
        """ save class to file """
        pickle.dump(self,open(datafile, 'w'))


    def plot(self,ax,
            colors=matplotlib.pyplot.cm.Reds_r,
            smooth=False,
            contour_levels='[1,2]',
            fine_contour_levels='numpy.arange(0, contour_levels[-1] + 1, fineness)',
            fineness=0.5,
            linewidths=1.0):

        # define the default contour lines as 1,2
        if contour_levels == '[1,2]':
            contour_levels = [1, 2]

        # Set up the fine contour gradation as 1 sigma above the levels above,
        # and with specified fineness
        if fine_contour_levels == 'numpy.arange(0, contour_levels[-1] + 1, fineness)':
            fine_contour_levels = numpy.arange(0, contour_levels[-1] + 1, fineness)

        # Create numpy arrays
        x = numpy.array(self.x)
        y = numpy.array(self.y)
        z = numpy.array(self.z)

        # Convert to sigmas
        z = numpy.sqrt(2) * erfinv(1 - z)

        # Gaussian filter if desired the sigmas by a factor of 1%
        if smooth:
            z = gaussian_filter(z, sigma=numpy.array(z.shape) / 100.0, order=0)

        # Plot the filled contours onto the axis ax
        for i in range(2):
            cbar = ax.contourf( x, y, z,
                cmap=colors, levels=fine_contour_levels)

        # Plot some sigma-based contour lines
        ax.contour( x, y, z,
            colors='k', linewidths=linewidths, levels=contour_levels)

        # Set limits on axes
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])

        # Return the contours for use as a colourbar later
        return cbar

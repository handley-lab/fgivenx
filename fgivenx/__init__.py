""" fgivenx module

    Methods
    -------
    - samples_from_getdist_chains
    - compute_contours
"""
import numpy
import scipy
import tqdm
import matplotlib.pyplot
from fgivenx.mass import PMF


def samples_from_getdist_chains(file_root, params):
    """ Extract samples and weights from getdist chains.

    Parameters
    ----------
    file_root: str
        Root name for getdist chains files. This script requires
        - file_root.txt
        - file_root.paramnames

    params: list(str)
        Names of parameters to be supplied to second argument of f(x|theta).

    Returns
    -------
    samples: numpy.array
        2D Array of samples. samples.shape=(# of samples, len(params),)

    weights: numpy.array
        Array of weights. samples.shape = (len(params),)
    """

    # Get the full data
    data = numpy.loadtxt(file_root + '.txt')
    weights = data[:, 0]

    # Get the paramnames
    paramnames = numpy.loadtxt(file_root + '.paramnames', dtype=str)
    if len(paramnames.shape) is 2:
        paramnames = paramnames[:, 0]

    # Get the relevant samples
    indices = [2+list(paramnames).index(p) for p in params]
    samples = data[:, indices]

    return samples, weights

def trim_samples(samples, weights, nsamp=0):

    n = len(weights)
    weights /= weights.max()
    choices = numpy.random.rand(n) < weights

    new_samples = samples[choices] 

    if nsamp > 0:
        new_samples = numpy.random.choice(new_samples)

    return new_samples

def compute_samples(f, x, samples):
    """ Apply f to x and theta.

    Parameters
    ----------
    See arguments of fgivenx.compute_contours

    Returns
    -------
    An array of samples at each x. shape=(len(x),len(samples))
    """
    return numpy.array([f(x,theta) for theta in tqdm.tqdm(samples)])


def compute_contours(f, x, samples, **kwargs):
    """ Compute the contours ready for matplotlib plotting.

    Parameters
    ----------
    f : function
        f(x|theta)

    x : array-like
        Descriptor of x values to evaluate.

    samples: numpy.array
        2D Array of theta samples. samples.shape=(# of samples, len(theta),)

    Keywords
    --------
    """

    if not len(samples.shape) is 2:
        raise ValueError("compute_contours: samples should be a 2D numpy array")

    x = numpy.array(x)

    weights = kwargs.pop('weights', None)
    ntrim = kwargs.pop('ntrim', 0)
    ny = kwargs.pop('ny', 100)

    if not weights is None:
        samples = trim_samples(samples, weights, ntrim)

    fsamples = compute_samples(f, x, samples)
    y = numpy.linspace(fsamples.min(), fsamples.max(), ny)
    z = numpy.array([PMF(s)(y) for s in tqdm.tqdm(fsamples.transpose())]).transpose()

    return x, y, z


def plot(x, y, z, ax, **kwargs):
    """ Plot computed contours.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot
            Axes to plot the contours onto.
            Typically generated with:
                fig, ax = matplotlib.pyplot.subplots()

        colors: matplotlib.colors.LinearSegmentedColormap, optional
            (Default: matplotlib.pyplot.cm.Reds_r)
            Color scheme to plot with. Recommend plotting in reverse
        smooth: bool, optional
            (Default: False)
            Whether to smooth the contours.
        contour_line_levels: List[float], optional
            (Default: [1,2])
            Contour lines to be plotted.
        linewidth: float, optional
            (Default: 0.1)
            Thickness of contour lines
        contour_color_levels: List[float], optional
            (Default: numpy.arange(0, contour_line_levels[-1] + 1, fineness))
            Contour color levels.
        fineness: float, optional
            (Default: 0.1)
            Spacing of contour color levels.
        x_trans: function: Float -> Float
            (Default: x->x)
            Function to transform the x coordinates by
        y_trans: function: Float -> Float
            (Default: y->y)
            Function to transform the y coordinates by
            

        Returns
        -------
        cbar: matplotlib.contour.QuadContourSet
            Colors to create a global colour bar

        Functionality mostly determined by modifications to ax
    """
    # Get inputs
    colors = kwargs.pop('colors', matplotlib.pyplot.cm.Reds_r)
    smooth = kwargs.pop('smooth', False)

    linewidths = kwargs.pop('linewidths', 1.0)
    contour_line_levels = kwargs.pop('contour_line_levels', [1, 2])

    fineness = kwargs.pop('fineness', 0.5)
    default_color_levels = numpy.arange(0, contour_line_levels[-1] + 1, fineness)
    contour_color_levels = kwargs.pop('contour_color_levels', default_color_levels)

    x_trans = kwargs.pop('x_trans', lambda x: x)
    x_trans = numpy.vectorize(x_trans)
    y_trans = kwargs.pop('y_trans', lambda y: y)
    y_trans = numpy.vectorize(y_trans)

    if kwargs:
        raise TypeError('Unexpected **kwargs in Contour plot method: %r' % kwargs)


    # Convert to sigmas
    z = numpy.sqrt(2) * scipy.special.erfinv(1 - z)

    # Gaussian filter if desired the sigmas by a factor of 1%
    if smooth:
        z = scipy.ndimage.gaussian_filter(z, sigma=numpy.array(z.shape) / 100.0, order=0)

    # Plot the filled contours onto the axis ax
    cbar = ax.contourf(x, y, z, cmap=colors, levels=contour_color_levels)

    # Remove those annoying white lines
    for c in cbar.collections:
        c.set_edgecolor("face")

    # Plot some sigma-based contour lines
    ax.contour(x, y, z, colors='k', linewidths=linewidths, levels=contour_line_levels)

    # Set limits on axes
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])

    # Return the contours for use as a colourbar later
    return cbar


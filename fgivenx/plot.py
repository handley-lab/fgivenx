import scipy
import scipy.ndimage
import numpy
import matplotlib.pyplot

def make_edges(x, keep_limits=True):
    r"""
    Convert a series of points to a set of bin edges with the points
    contained within the bins.

    Turn the List
    X__X____X__X____X____X____X__X____X
    into the edges
    |_|___|___|___|____|____|___|___|__|

    Parameters
    ----------
    x: numpy array
        List of points

    keep_limits: bool, optional
        Whether to make sure the edges are within [min(x), max(x)].
        If True the first and last bin are smaller to stay within these
        boundaries, if False they will stretch beyond the limits.
        Default: True

    Returns
    -------
    edges: numpy array
        bin edges with length `len(x)+1`
    """
    diff = numpy.diff(x)
    edges = x[:-1]+diff/2
    if keep_limits:
        edges = numpy.append(edges, x[-1])
        edges = numpy.insert(edges, 0, x[0])
    else:
        edges = numpy.append(edges, x[-1]+diff[-1]/2)
        edges = numpy.insert(edges, 0, x[0]-diff[0]/2)
    return edges


def plot(x, y, z, ax=None, **kwargs):
    r"""
    Plot iso-probability mass function, converted to sigmas.

    Parameters
    ----------
    x, y, z : numpy arrays
        Same as arguments to :func:`matplotlib.pyplot.contour`, or for
        histogram to :func:`matplotlib.pyplot.pcolormesh`

    ax: axes object, optional
        :class:`matplotlib.axes._subplots.AxesSubplot` to plot the contours
        onto. If unsupplied, then :func:`matplotlib.pyplot.gca()` is used to
        get the last axis used, or create a new one.

    colors: color scheme, optional
        :class:`matplotlib.colors.LinearSegmentedColormap`
        Color scheme to plot with. Recommend plotting in reverse
        (Default: :class:`matplotlib.pyplot.cm.Reds_r`)

    linecolors: color string or sequence of colors, optional
        Colors for contour lines (Default: 'k')

    histogram: bool, optional
        Replaces the contourf (filling of contours) plot by a histogram.
        (Default: False)

    pdf_histogram: bool, optional:
        When plotting histogram, interpret z as a PDF instead of
        a PMF (plotted as "sigmas"). Default: `False`

    pdf_histogram_norm: `~matplotlib.colors.Normalize`, optional
        Normalization for histogram color map if plotting the PDF.
        (i.e. only effective if `histogram and pdf_histogram`). E.g. a
        logarithmic norm can be obtained from `matplotlib.colors.LogNorm()`.
        Default: `None`

    alpha: float, optional
        Transparency of filled contours. Given as alpha blending
        value between 0 (transparent) and 1 (opague).

    smooth: float, optional
        Percentage by which to smooth the contours. Not recommended when
        using `histogram`. (Default: no smoothing)

    contour_line_levels: List[float], optional
        Contour lines to be plotted.  (Default: [1,2])

    linewidths: float, optional
        Thickness of contour lines.  (Default: 0.3)

    contour_color_levels: List[float], optional
        Contour color levels.
        (Default: `numpy.arange(0, contour_line_levels[-1] + 1, fineness)`)

    fineness: float, optional
        Spacing of contour color levels.  (Default: 0.1)

    lines: bool, optional
        (Default: True)

    rasterize_contours: bool, optional
        Rasterize the contours while keeping the lines, text etc in vector
        format. Useful for reducing file size bloat and making printing
        easier when you have dense contours.
        (Default: False)

    Returns
    -------
    cbar: color bar
        :class:`matplotlib.contour.QuadContourSet`
        Colors to create a global colour bar
    """
    if ax is None:
        ax = matplotlib.pyplot.gca()
    # Get inputs
    colors = kwargs.pop('colors', matplotlib.pyplot.cm.Reds_r)
    linecolors = kwargs.pop('linecolors', 'k')

    histogram = kwargs.pop('histogram', False)
    pdf_histogram = kwargs.pop('pdf_histogram', None)
    pdf_histogram_norm = kwargs.pop('pdf_histogram_norm', None)

    smooth = kwargs.pop('smooth', False)
    if smooth and histogram:
        print("WARNING: You selected smooth and histogram,"
              "are you sure you want to plot a smoothed histogram?")

    linewidths = kwargs.pop('linewidths', 0.3)
    contour_line_levels = kwargs.pop('contour_line_levels', [1, 2, 3])

    fineness = kwargs.pop('fineness', 0.5)
    default_color_levels = numpy.arange(0, contour_line_levels[-1] + 1,
                                        fineness)
    contour_color_levels = kwargs.pop('contour_color_levels',
                                      default_color_levels)

    rasterize_contours = kwargs.pop('rasterize_contours', False)

    alpha = kwargs.pop('alpha', 1)

    lines = kwargs.pop('lines', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if histogram and pdf_histogram:
        # cmap is reversed for easy compatibility with histogram = False
        cbar = ax.pcolormesh(make_edges(x), y, z, cmap=colors.reversed(), alpha=alpha,
                             norm=pdf_histogram_norm)


    # Convert to sigmas
    z = numpy.sqrt(2) * scipy.special.erfinv(1 - z)

    # Gaussian filter if desired the sigmas by a factor of smooth%
    if smooth:
        sigma = smooth*numpy.array(z.shape)/100.0
        z = scipy.ndimage.gaussian_filter(z, sigma=sigma, order=0)

    if histogram and not pdf_histogram:
        cbar = ax.pcolormesh(make_edges(x), y, z, cmap=colors, alpha=alpha, vmin=numpy.min(contour_color_levels), vmax=numpy.max(contour_color_levels))

    # Plot the filled contours onto the axis ax
    if not histogram:
        cbar = ax.contourf(x, y, z, cmap=colors, levels=contour_color_levels,
                       alpha=alpha)

        # Rasterize contours (the rest of the figure stays in vector format)
        if rasterize_contours:
            for c in cbar.collections:
                c.set_rasterized(True)

        # Remove those annoying white lines
        for c in cbar.collections:
            c.set_edgecolor("face")

    # Plot some sigma-based contour lines
    if lines and not histogram:
        ax.contour(x, y, z, colors=linecolors, linewidths=linewidths,
                   levels=contour_line_levels)

    # Return the contours for use as a colourbar later
    return cbar


def plot_lines(x, fsamps, ax=None, downsample=100, **kwargs):
    """
    Plot function samples as a set of line plots.

    Parameters
    ----------
    x: 1D array-like
        x values to plot

    fsamps: 2D array-like
        set of functions to plot at each x. As returned by
        :func:`fgivenx.compute_samples`

    ax: axes object
        :class:`matplotlib.pyplot.ax` to plot on.

    downsample: int, optional
        Reduce the number of samples to a viewable quantity. (Default 100)

    any other keywords are passed to :meth:`matplotlib.pyplot.ax.plot`
    """
    if ax is None:
        ax = matplotlib.pyplot.gca()
    if downsample < len(fsamps.T):
        indices = numpy.random.choice(len(fsamps.T), downsample, replace=False)
    else:
        indices = numpy.arange(len(fsamps.T))
    color = kwargs.pop('color', 'k')
    alpha = kwargs.pop('alpha', 0.1)
    for y in fsamps.T[indices]:
        ax.plot(x, y, color=color, alpha=alpha, **kwargs)

import scipy
import scipy.ndimage
import numpy
import matplotlib.pyplot


def plot(x, y, z, ax=None, **kwargs):
    r"""
    Plot iso-probability mass function, converted to sigmas.

    Parameters
    ----------
    x, y, z : numpy arrays
        Same as arguments to :func:`matplotlib.pyplot.contour`

    ax: axes object, optional
        :class:`matplotlib.axes._subplots.AxesSubplot` to plot the contours
        onto. If unsupplied, then :func:`matplotlib.pyplot.gca()` is used to
        get the last axis used, or create a new one.

    colors: color scheme, optional
        :class:`matplotlib.colors.LinearSegmentedColormap`
        Color scheme to plot with. Recommend plotting in reverse
        (Default: :class:`matplotlib.pyplot.cm.Reds_r`)

    alpha: float, optional
        Transparency of filled contours. Given as alpha blending
        value between 0 (transparent) and 1 (opague).

    smooth: float, optional
        Percentage by which to smooth the contours.
        (Default: no smoothing)

    contour_line_levels: List[float], optional
        Contour lines to be plotted.  (Default: [1,2])

    linewidths: float, optional
        Thickness of contour lines.  (Default: 0.3)

    contour_color_levels: List[float], optional
        Contour color levels. (Default:
        `numpy.arange(0, contour_line_levels[-1] + fineness / 2, fineness)`)

    fineness: float, optional
        Spacing of contour color levels.  (Default: 0.5)

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
    smooth = kwargs.pop('smooth', False)

    linewidths = kwargs.pop('linewidths', 0.3)
    contour_line_levels = kwargs.pop('contour_line_levels', [1, 2, 3])

    fineness = kwargs.pop('fineness', 0.5)
    default_color_levels = numpy.arange(0,
                                        contour_line_levels[-1] + fineness / 2,
                                        fineness)
    contour_color_levels = kwargs.pop('contour_color_levels',
                                      default_color_levels)

    rasterize_contours = kwargs.pop('rasterize_contours', False)

    alpha = kwargs.pop('alpha', 1)

    lines = kwargs.pop('lines', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # Convert to sigmas
    z = numpy.sqrt(2) * scipy.special.erfinv(1 - z)

    # Gaussian filter if desired the sigmas by a factor of smooth%
    if smooth:
        sigma = smooth*numpy.array(z.shape)/100.0
        z = scipy.ndimage.gaussian_filter(z, sigma=sigma, order=0)

    # Plot the filled contours onto the axis ax
    cbar = ax.contourf(x, y, z, cmap=colors, levels=contour_color_levels,
                       alpha=alpha)

    # Rasterize contours (the rest of the figure stays in vector format)
    if rasterize_contours:
        for c in cbar.collections:
            c.set_rasterized(True)

    # Remove those annoying white lines
    cbar.set_edgecolor("face")

    # Plot some sigma-based contour lines
    if lines:
        ax.contour(x, y, z, colors='k', linewidths=linewidths,
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

import numpy
import pytest
from fgivenx.plot import plot, plot_lines
import matplotlib.pyplot as plt
import matplotlib


def test_plot():
    nx = 100
    ny = 101
    x = numpy.linspace(0, 1, nx)
    y = numpy.linspace(0, 1, ny)
    z = numpy.random.rand(ny, nx)

    fig, ax = plt.subplots()
    cbar = plot(x, y, z, ax)
    assert type(cbar) is matplotlib.contour.QuadContourSet

    with pytest.raises(TypeError):
        plot(x, y, z, ax, wrong_argument=None)

    cbar = plot(x, y, z, ax, smooth=1)
    cbar = plot(x, y, z, ax, rasterize_contours=True)
    cbar = plot(x, y, z, ax, lines=False)


def test_plot_lines():
    nx = 100
    nsamps = 150
    x = numpy.linspace(0, 1, nx)
    m = numpy.random.normal(nsamps)
    c = numpy.random.normal(nsamps)
    fsamps = numpy.outer(x, m) + c
    fig, ax = plt.subplots()
    plot_lines(x, fsamps, ax)
    plot_lines(x, fsamps, ax, downsample=200)

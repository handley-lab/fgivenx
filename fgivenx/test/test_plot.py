import numpy
import pytest
from fgivenx.plot import plot, plot_lines
import matplotlib.pyplot as plt
import matplotlib


def gen_plot_data():
    numpy.random.seed(0)
    nx = 100
    ny = 101
    x = numpy.linspace(0, 1, nx)
    y = numpy.linspace(0, 1, ny)
    X, Y = numpy.meshgrid(x, y)
    z = numpy.exp(-((X-0.5)**2+(Y-0.5)**2)/0.01)
    return x, y, z


def test_plot():
    x, y, z = gen_plot_data()

    fig, ax = plt.subplots()
    cbar = plot(x, y, z, ax)
    assert type(cbar) is matplotlib.contour.QuadContourSet


def test_plot_wrong_argument():
    x, y, z = gen_plot_data()
    with pytest.raises(TypeError):
        plot(x, y, z, wrong_argument=None)


def test_plot_no_ax():
    plt.subplots()
    x, y, z = gen_plot_data()
    plot(x, y, z)


def test_plot_smooth():
    plt.subplots()
    x, y, z = gen_plot_data()
    plot(x, y, z, smooth=1)


def test_plot_rasterize():
    plt.subplots()
    x, y, z = gen_plot_data()
    plot(x, y, z, rasterize_contours=True)


def test_plot_nolines():
    plt.subplots()
    x, y, z = gen_plot_data()
    plot(x, y, z, lines=False)


def gen_line_data():
    numpy.random.seed(0)
    nx = 100
    nsamps = 150
    x = numpy.linspace(0, 1, nx)
    m = numpy.random.normal(size=nsamps)
    c = numpy.random.normal(size=nsamps)
    fsamps = numpy.outer(x, m) + c
    return x, fsamps


def test_plot_lines():
    x, fsamps = gen_line_data()
    fig, ax = plt.subplots()
    plot_lines(x, fsamps, ax)


def test_plot_lines_no_ax():
    x, fsamps = gen_line_data()
    plt.subplots()
    plot_lines(x, fsamps)


def test_plot_lines_downsample():
    x, fsamps = gen_line_data()
    plt.subplots()
    plot_lines(x, fsamps, downsample=200)

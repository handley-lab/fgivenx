import numpy
import pytest
import os
import fgivenx.io
import pytest
from shutil import rmtree
from numpy.testing import assert_allclose, assert_array_equal
import scipy.stats
import scipy.integrate
import scipy.special
from fgivenx.samples import compute_samples, samples_from_getdist_chains
import getdist

def test_compute_samples():

    with pytest.raises(TypeError):
        compute_samples(None,None,None,wrong_argument=None)

    cache = '.test_cache/test'
    numpy.random.seed(0)
    nsamp = 5000
    a,b,e,f = 0,1,0,1
    m = numpy.random.normal(a,b,nsamp)
    c = numpy.random.normal(e,f,nsamp)
    samples = numpy.array([list(_) for _ in zip(m,c)])
    nx = 100
    x = numpy.linspace(-1,1,nx)
    fsamps_ = numpy.outer(x, m) + c
    def f(x, theta):
        m, c = theta
        return m*x+c

    assert(not os.path.isfile(cache + '_fsamples.pkl'))
    fsamps = compute_samples([f],x,[samples], cache=cache)
    assert(os.path.isfile(cache + '_fsamples.pkl'))

    assert_allclose(fsamps_, fsamps,)

    fsamps = compute_samples([f],x,[samples], cache=cache)
    assert_allclose(fsamps_, fsamps,)

    rmtree('.test_cache')


def test_samples_from_getdist_chains():

    # Set up getdist chains
    file_root = './chains/test' 
    nsamples = 1000
    labels = [r'\alpha',r'\beta',r'\gamma']
    names = ['a', 'b', 'g']
    samples_ = numpy.random.rand(nsamples,len(names))
    weights_ = numpy.random.rand(nsamples)
    samples = getdist.mcsamples.MCSamples(samples=samples_,labels=labels,names=names,weights=weights_)
    samples.saveAsText(file_root,make_dirs=True)

    # now test function
    params = ['a','g']

    chains_file = file_root + '.txt'
    paramnames_file = file_root + '.paramnames'
    with pytest.raises(ValueError):
        samples_from_getdist_chains(params)
    with pytest.raises(ValueError):
        samples_from_getdist_chains(params, chains_file=chains_file)
    with pytest.raises(ValueError):
        samples_from_getdist_chains(params, paramnames_file=paramnames_file)

    i = [names.index(p) for p in params]

    samples, weights = samples_from_getdist_chains(params, 
                                                   chains_file=chains_file, 
                                                   paramnames_file=paramnames_file)
    assert_allclose(samples,samples_[:,i])
    assert_allclose(weights,weights_)

    samples, weights = samples_from_getdist_chains(params, file_root=file_root) 
    assert_allclose(samples,samples_[:,i])
    assert_allclose(weights,weights_)

    samples, weights, latex = samples_from_getdist_chains(params, file_root=file_root, latex=True) 
    assert_allclose(weights,weights_)
    assert_array_equal(latex, numpy.array(labels)[i])

    rmtree('chains')



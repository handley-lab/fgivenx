import numpy
import pytest
import os
import fgivenx.io
import pytest
from shutil import rmtree
from numpy.testing import assert_allclose

def test_Cache():
    """ Fully test the cache """

    # Stuff for testing the functionality
    numpy.random.seed(0)
    dirname = '.test_cache'
    root = os.path.join(dirname,'test_root')
    data0 = numpy.random.rand(5,4,3)
    data1 = numpy.random.rand(4,3)
    data2 = data0 * data1

    # Creates a directory
    assert(not os.path.isdir(dirname))
    cache = fgivenx.io.Cache(root)
    assert(os.path.isdir(dirname))

    # Raises exception if there's no cache
    with pytest.raises(fgivenx.io.CacheMissing):
        cache.load()

    # Check saving
    cache.save(data0, data1, data2)
    assert(os.path.exists(root + '.pkl'))

    # check loading
    data0_, data1_, data2_ = cache.load()
    assert_allclose(data0_,data0)
    assert_allclose(data1_,data1)
    assert_allclose(data2_,data2)

    # check check
    with pytest.raises(ValueError):
        cache.check(data0)

    with pytest.raises(fgivenx.io.CacheChanged):
        cache.check(data1, data0)

    with pytest.raises(fgivenx.io.CacheChanged):
        cache.check(numpy.random.rand(5,4,3),numpy.random.rand(4,3))

    assert_allclose(data2, cache.check(data0, data1))

    cache.save([data0, data1], data2)

    assert_allclose(data2, cache.check([data0_, data1_]))

    with pytest.raises(fgivenx.io.CacheChanged):
        cache.check([data1])

    with pytest.raises(fgivenx.io.CacheChanged):
        cache.check([data1, data0])

    with pytest.raises(fgivenx.io.CacheChanged):
        cache.check([data0, 1+data1])

    # Remove the testing cache
    rmtree(dirname)

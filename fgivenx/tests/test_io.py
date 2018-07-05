import numpy
import pytest
import os
import fgivenx.io
import pytest

def test_Cache_dir_creation():
    """ Check that cache can create directories """
    dirname = '.test_cache'
    root = os.path.join(dirname,'test_root')
    assert(not os.path.isdir(dirname))
    cache = fgivenx.io.Cache(root)
    assert(os.path.isdir(dirname))
    os.rmdir(dirname)

def test_Cache_load():
    """ Check that cache """

    root = '.test_cache/test_root'
    cache = fgivenx.io.Cache(root)
    with pytest.raises(fgivenx.io.CacheMissing):
        cache.load()
    os.rmdir('.test_cache')

import numpy
import pytest
import os
from numpy.testing import assert_allclose

import fgivenx.io

def test_cache_dir_creation():
    dirname = '.test_cache'
    root = os.path.join(dirname,'test_root')
    assert(not os.path.isdir(dirname))
    cache = fgivenx.io.Cache(root)
    assert(os.path.isdir(dirname))
    os.rmdir(dirname)

import numpy
import os
import pytest
from shutil import rmtree
from numpy.testing import assert_allclose, assert_almost_equal
from fgivenx.parallel import parallel_apply

def f(x):
    return x**3

def test_parallel_apply():

    array = numpy.linspace(-1,1,100)

    with pytest.raises(TypeError):
        parallel_apply(f,array,wrong_argument=None)

    with pytest.raises(ValueError):
        parallel_apply(f,array,parallel='wrong_argument')

    def f_local(x):
        return x**3

    with pytest.raises(AttributeError):
        parallel_apply(f_local,array, parallel=True)

    farray = parallel_apply(f_local,array,parallel=False)
    assert_allclose([f(x) for x in array],farray)

    for par in [False, True, -1, 2]:
        farray = parallel_apply(f,array,parallel=par)
        assert_allclose([f(x) for x in array],farray)


    
    

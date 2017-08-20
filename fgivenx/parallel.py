import tqdm
import os
import numpy
from joblib import Parallel, delayed
from mpi4py import MPI


def openmp_apply(f, array, **kwargs):
    """ Apply a function to an array with openmp parallelisation.

    Equivalent to [f(x) for x in array], but parallelised. Will parallelise
    using the environment variable OMP_NUM_THREADS, unless nprocs is provided.

    Parameters
    ----------
    f: function
        Univariate function to apply to each element of array

    array: array-like
        Array to apply f to

    Keywords
    --------
    nprocs: int
        Force to parallelise with nprocs.

    precurry: tuple
        arguments to pass to f before 

    postcurry: tuple
        arguments to pass to f after 
    """

    precurry = kwargs.pop('precurry',())
    postcurry = kwargs.pop('postcurry',())
    nprocs = kwargs.pop('nprocs',None)

    if nprocs is None:
        try:
            nprocs = int(os.environ['OMP_NUM_THREADS'])
            if nprocs is 1:
                print("Warning: You have requested to use openmp, but environment"
                      "variable OMP_NUM_THREADS=1")
        except KeyError:
            raise EnvironmentError(
                    "You have requested to use openmp, but the environment"
                    "variable OMP_NUM_THREADS is not set")


    return Parallel(n_jobs=nprocs)(
                                   delayed(f)(*precurry,x,*postcurry) for x in tqdm.tqdm(array)
                                  )


def mpi_apply(function, array, **kwargs):
    """ Distribute a function applied to an array across an MPI communicator

    Parameters
    ----------
    function:
        function maps x -> y where x and y are numpy ND arrays, and the
        dimensionality of x is determined by xdims

    array:
        ndarray to apply function to

    Keywords
    --------
    comm:
        MPI communicator. If not supplied, one will be created
    """

    if not MPI.Is_initialized():
        MPI.Init()
    comm = kwargs.pop('comm', MPI.COMM_WORLD)

    array_local = mpi_scatter_array(array, comm)

    if comm.Get_rank() is 0:
        print("rank 0")
        array_local = tqdm.tqdm(array_local)

    answer_local = numpy.array([function(x) for x in array_local])

    return mpi_gather_array(answer_local, comm)


def mpi_scatter_array(array, comm):
    """ Scatters an array across all processes across the first axis"""
    array = array.astype('d').copy()

    rank = comm.Get_rank()
    n = len(array)
    nprocs = comm.Get_size()

    sendcounts = numpy.empty(nprocs,dtype='int')
    sendcounts.fill(n//nprocs)
    sendcounts[:n-sum(sendcounts)] += 1

    displacements = sendcounts.cumsum() - sendcounts

    shape = array.shape

    sendcount = sendcounts[rank]
    array_local = numpy.zeros((sendcount,) + shape[1:])

    sendcounts *= numpy.prod(shape[1:])
    displacements *= numpy.prod(shape[1:])

    comm.Scatterv([array, sendcounts, displacements, MPI.DOUBLE], array_local)
    return array_local


def mpi_gather_array(array_local, comm):
    """ Gathers an array from all processes"""
    shape = array_local.shape

    sendcounts = numpy.array(comm.allgather(len(array_local)))
    displacements = sendcounts.cumsum() - sendcounts

    array = numpy.zeros((sum(sendcounts),) + shape[1:])
    sendcounts *= numpy.prod(shape[1:])
    displacements *= numpy.prod(shape[1:])

    comm.Allgatherv(array_local, [array, sendcounts, displacements, MPI.DOUBLE])
    return array

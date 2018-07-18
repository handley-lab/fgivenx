import os
import pickle
import numpy
import inspect


class CacheException(Exception):
    """ Base exception to indicate cache errors """

    def calling_function(self):
        """ Get the name of the function calling this cache. """
        return inspect.getouterframes(inspect.currentframe())[3][3]

    def __str__(self):
        """ Return the cache message. """
        return self._msg


class CacheOK(CacheException):
    """ Exception to indicate the cache can be used. """
    def __init__(self, file_root):
        self._msg = "%s: reading from cache in %s" % (
                     self.calling_function(), file_root)


class CacheChanged(CacheException):
    """ Exception to indicate the cache has changed. """
    def __init__(self, file_root):
        self._msg = "%s: values have changed in cache %s, recomputing" % (
                     self.calling_function(), file_root)


class CacheMissing(CacheException):
    """ Exception to indicate the cache does not exist. """
    def __init__(self, file_root):
        self._msg = "%s: No cache file %s" % (
                     self.calling_function(), file_root)


class Cache(object):
    """ Cacheing tool.

    Parameters
    ----------
    file_root: str
        cached values are saved in file_root.pkl
    """
    def __init__(self, file_root):
        self.file_root = file_root
        dirname = os.path.dirname(self.file_root)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def check(self, *args):
        """ Check that the cache has changed.

        Parameters
        ----------
        *args:
            All but the last argument are inputs to the cached function. The
            last is the actual value of the function.

        Returns
        -------
        If cache is unchanged, return the last answer, otherwise indicate
        recomputation required by throwing a :class:`CacheException`.
        """
        data = self.load()

        if len(data)-1 != len(args):
            raise ValueError("Wrong number of arguments passed to Cache.check")

        try:
            for x, x_check in zip(data, args):
                if isinstance(x, list):
                    if len(x) != len(x_check):
                        raise CacheException
                    for x_i, x_check_i in zip(x, x_check):
                        if x_i.shape != x_check_i.shape:
                            raise CacheException
                        elif not numpy.allclose(x_i, x_check_i,
                                                equal_nan=True):
                            raise CacheException
                elif x.shape != x_check.shape:
                    raise CacheException
                elif not numpy.allclose(x, x_check, equal_nan=True):
                    raise CacheException

        except CacheException:
            raise CacheChanged(self.file_root)

        print(CacheOK(self.file_root))
        return data[-1]

    def load(self):
        """ Load cache from file using pickle. """
        try:
            with open(self.file_root + '.pkl', "rb") as f:
                return pickle.load(f)
        except IOError:
            raise CacheMissing(self.file_root)

    def save(self, *args):
        """ Save cache to file using pickle.

        Parameters
        ----------
        *args:
            All but the last argument are inputs to the cached function. The
            last is the actual value of the function.
        """
        with open(self.file_root + '.pkl', "wb") as f:
            pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)

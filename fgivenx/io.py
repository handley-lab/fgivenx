import os
import pickle
import errno
import numpy
import inspect

class CacheError(Exception):
    """ Base exception to indicate cache errors """
    def __init__(self, file_root):
        self._msg = "%s: reading from cache in %s" % (self.calling_function(), file_root)
    def calling_function(self):
        return inspect.getouterframes(inspect.currentframe())[3][3]
    def msg(self):
        return self._msg


class CacheChanged(CacheError):
    def __init__(self, file_root):
        self._msg = "%s: values have changed in cache %s, recomputing" % (self.calling_function(), file_root)


class CacheMissing(CacheError):
    def __init__(self, file_root):
        self._msg = "%s: No cache file %s" % (self.calling_function(), file_root)


class Cache(object):

    def __init__(self, file_root):
        if isinstance(file_root, Cache):
            self.file_root = file_root.file_root
        else:
            self.file_root = file_root

        dirname = os.path.dirname(self.file_root)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    def check(self, *args):

        data = self.load()
        if len(data)-1 != len(args):
            raise ValueError("Wrong number of arguments passed to Cache.check")

        try:
            for x, x_check in zip(data, args):
                if isinstance(x, list):
                    if len(x) != len(x_check):
                        raise CacheError
                    for x_i, x_check_i in zip(x, x_check):
                        if x_i.shape != x_check_i.shape:
                            raise CacheError
                        elif not numpy.allclose(x_i, x_check_i,equal_nan=True):
                            raise CacheError
                elif x.shape != x_check.shape:
                    raise CacheError
                elif not numpy.allclose(x,x_check,equal_nan=True):
                    raise CacheError
        except CacheError:
            raise CacheChanged(self.file_root)

        print(CacheError(self.file_root).msg())
        return data[-1]

    def load(self):
        try:
            with open(self.file_root + '.pkl', "rb") as f:
                return pickle.load(f)
        except IOError:
            raise CacheMissing(self.file_root)
    
    def save(self, *args):
        with open(self.file_root + '.pkl', "wb") as f:
            pickle.dump(args, f,protocol=pickle.HIGHEST_PROTOCOL)

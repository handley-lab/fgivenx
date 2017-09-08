import os
import pickle
import errno
import numpy
import inspect

class CacheError(IOError):
    pass

class CacheFile(object):
    def __get__(self, obj, type=None):
        try:
            with open(self.filename(obj),"rb") as f:
                return pickle.load(f)
        except IOError:

            calling_function = inspect.getouterframes(inspect.currentframe())[2][3]
            raise CacheError(calling_function + ": No cache file %s" % obj.file_root)
    
    def __set__(self, obj, value):
        with open(self.filename(obj),"wb") as f:
            pickle.dump(value, f,protocol=pickle.HIGHEST_PROTOCOL)

    def __delete__(self, obj):
        try:
            os.remove(self.filename(obj))
        except OSError:
            pass

    def filename(self, obj):
        return obj.file_root + '.pkl'

    def dirname(self,obj):
        return os.path.dirname(self.filename(obj))

class Cache(object):

    data = CacheFile()

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

        calling_function = inspect.getouterframes(inspect.currentframe())[2][3]

        if len(self.data)-1 != len(args):
            raise ValueError("Wrong number of arguments passed to Cache.check")

        for x, x_check in zip(self.data, args):
            if isinstance(x, list):
                if len(x) != len(x_check):
                    raise CacheError(calling_function + ": values have changed in cache %s, recomputing" % self.file_root)
                for x_i, x_check_i in zip(x, x_check):
                    if not numpy.array_equal(x_i, x_check_i):
                        raise CacheError(calling_function + ": values have changed in cache %s, recomputing" % self.file_root )

            elif not numpy.array_equal(x,x_check):
                raise CacheError(calling_function + ": values have changed in cache %s, recomputing" % self.file_root )

        print(calling_function + ": reading from cache in %s" % self.file_root)
        return self.data[-1]

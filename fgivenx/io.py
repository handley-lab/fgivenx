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

            calling_function = inspect.getouterframes(inspect.currentframe())[1][3]
            raise CacheError(calling_function + ": No cache file present")
    
    def __set__(self, obj, value):
        with open(self.filename(obj),"wb") as f:
            pickle.dump(value, f)

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

        calling_function = inspect.getouterframes(inspect.currentframe())[1][3]

        if len(self.data)-1 != len(args):
            raise ValueError("Wrong number of arguments passed to Cache.check")

        for x, x_check in zip(self.data, args):
            if not numpy.array_equal(x,x_check):
                raise CacheError(calling_function + ": values have changed, recomputing")

        print(calling_function + ": reading from cache")
        return self.data[-1]

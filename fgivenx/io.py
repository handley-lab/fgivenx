import os
import pickle
import errno

class CacheError(IOError):
    pass

class CacheFile(object):
    def __init__(self, extension):
        self.extension = extension

    def __get__(self, obj, type=None):
        try:
            with open(self.filename(obj),"rb") as f:
                return pickle.load(f)
        except IOError:
            raise CacheError
    
    def __set__(self, obj, value):
        with open(self.filename(obj),"wb") as f:
            pickle.dump(value, f)

    def __delete__(self, obj):
        os.remove(self.filename(obj))

    def filename(self, obj):
        return obj.file_root + self.extension

    def dirname(self,obj):
        return os.path.dirname(self.filename(obj))


class Cache(object):
    fsamps = CacheFile('_fsamps.pkl')
    masses = CacheFile('_masses.pkl')

    def __init__(self, file_root):
        self.file_root = file_root

        dirname = os.path.dirname(self.file_root)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

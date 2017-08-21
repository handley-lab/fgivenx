import os
import pickle

def makedirs(dirname):
    """ Create a directory if it doesn't exist, avoiding race conditions."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

class CacheError(FileNotFoundError):
    pass

class Cache(object):
    def __init__(self, file_root):
        if file_root is None:
            return None
        self.file_root = file_root
        makedirs(os.path.dirname(self.file_root))

    @property
    def fsamps_filename(self):
        return self.file_root + '_fsamps.pkl'

    @property
    def masses_filename(self):
        return self.file_root + '_masses.pkl'

    @property
    def fsamps(self):
        return self.read(self.fsamps_filename)

    @fsamps.setter
    def fsamps(self, value):
        self.write(value, self.fsamps_filename)

    @fsamps.deleter
    def fsamps(self):
        self.clear(self.fsamps_filename)



    @property
    def masses(self):
        return self.read(self.masses_filename)

    @masses.setter
    def masses(self, value):
        self.write(value, self.masses_filename)

    @masses.deleter
    def masses(self):
        self.clear(self.masses_filename)

    def write(self, obj, filename):
        with open(filename,"wb") as f:
            pickle.dump(obj,f)

    def read(self, filename):
        try:
            with open(filename,"rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise CacheError

    def clear(self, filename):
        os.remove(filename)

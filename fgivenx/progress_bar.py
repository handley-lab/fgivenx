""" Wrapper module for progress bar.

    Checks to see if tqdm is installed on system, if it isn't use the basic
    version included in fgivenx.

"""
try:
    from tqdm import tqdm
except ImportError:
    from fgivenx.tqdm_simple.tqdm import tqdm

def pbar(iterator, desc=None):
    """ Wrapper function for progress bar """
    return tqdm(iterator, desc=desc)

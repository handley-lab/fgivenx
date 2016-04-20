try:
    from tqdm import tqdm
except ImportError:
    from tqdm_simple.tqdm import tqdm

def pbar(iterator,desc=None):
    if desc is None:
        return tqdm(iterator)
    else:
        return tqdm(iterator,desc=desc)


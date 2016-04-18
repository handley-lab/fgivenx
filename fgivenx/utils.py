from itertools import tee, izip
from scipy.optimize import brentq as root_finder

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def find_all_roots(xs,ys,y,function):

    def f(x):
        return function(x)-y

    roots = []
    for (x0, x1), (y0, y1) in zip(pairwise(xs), pairwise(ys)):
        if y0 == y:
            roots.append(x0)
        elif (y0 < y and y1 > y) or (y0 > y and y1 < y):
            x = root_finder(f,x0,x1)
            roots.append(x)
            
    return roots
            


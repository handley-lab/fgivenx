# SciPy kernel density estimator
from scipy.stats import gaussian_kde
def scipy_compute_kernel(data):
    return gaussian_kde(data)

def scipy_pdf(y,kernel):
    return kernel.evaluate(y)


# statsmodels kernel density estimator
import statsmodels.nonparametric.kde as KDETOOLS
def statsmodels_compute_kernel(data):
    kde = KDETOOLS.KDEUnivariate(data)
    kde.fit()
    return kde

def statsmodels_pdf(y,kernel):
    return kernel.evaluate(y)


# Choose the kernel
def compute_kernel(data):
    return scipy_compute_kernel(data)

def pdf(y,kernel):
    return scipy_pdf(y,kernel)

from scipy.interpolate import interp1d
from numpy import concatenate,sort,log,exp,isnan
def fast_kernel(data,ys):
    n      = ys.size
    kernel = compute_kernel(data)

    points = sort(
            concatenate(
                ( kernel.resample(n)[0] , ys )
                )
            )

    probs  = pdf(points,kernel)

    probs[probs==0] = 1e-300
    logprobs  = log(probs)
    return interp1d(points,logprobs,kind='quadratic',bounds_error=False,fill_value=log(1e-300))

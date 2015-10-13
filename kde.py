# SciPy kernel density estimator
from scipy.stats import gaussian_kde
def scipy_compute_kernel(data):
    return gaussian_kde(data)

def scipy_pdf(x,kernel):
    return kernel.evaluate(x)


# statsmodels kernel density estimator
import statsmodels.nonparametric.kde as KDETOOLS
def statsmodels_compute_kernel(data):
    kde = KDETOOLS.KDEUnivariate(data)
    kde.fit()
    return kde

def statsmodels_pdf(x,kernel):
    return kernel.evaluate(x)


# Choose the kernel
def compute_kernel(data):
    return scipy_compute_kernel(data)

def pdf(x,kernel):
    return scipy_pdf(x,kernel)

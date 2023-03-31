# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:10:43 2022

@author: DU

Using Cochrane-Orcutt Procedure to estimate 
the slope and intercept of autocorrelated data

Reference: Cochrane, D., & Orcutt, G. H. (1949). 
            Application of least squares regression to relationships containing auto-correlated error terms.
            Journal of the American Statistical Association, 44(245), 32–61. 
            https://doi.org/10.1080/01621459.1949.10483290
"""

import numpy as np
from scipy import signal
from scipy import stats

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def Cochrane_Orcutt(yt, xt, mean = False):
    """
    Calculate a linear least-squares regression for two sets of measurements,
        which is auto-correlated via the Cochrane-Orcutt Procedure.

    Parameters
    ----------
    yt, xt : array_like
        Two sets of measurements. Both arrays should have the same length.
    mean: Logic, default is False
        if True, returned intercept is the mean value of the regression line,
        else, returns the intercept.
        
    Returns
    -------
    result : dotdict
        The return value is an object with the following attributes:
        slope : float
            Slope of the regression line.
        intercept : float
            Mean value of the regression line.
        rvalue : float
            Correlation coefficient.
        pvalue : float
            The p-value for a hypothesis test whose null hypothesis is
            that the slope is zero, using Wald Test with t-distribution of
            the test statistic. See `alternative` above for alternative
            hypotheses.
        slope_se : float
            Standard error of the estimated slope.
        intercept_se : float
            Standard error of the estimated intercept.
    
    See Also
    --------
    scipy.stats.linregress
        Calculate a linear least-squares regression for two sets of measurements.
    
    Examples
    --------
    import numpy as np
    from scipy import signal
    from scipy import stats
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng()
    x = np.linspace(0, 10, 10)
    y = 1.6*x + rng.random(10)
    res = Cochrane_Orcutt(y, x, False)
    
    plt.plot(x, y)
    plt.plot(x, res.slope * x + res.intercept)    
    """
    
    if mean:
        xt = xt - xt.mean()
    epsilon = signal.detrend(yt)
    r = np.corrcoef(epsilon[1::], epsilon[0:-1])[0, 1]
    result = stats.linregress(xt, yt)
    regress = result.slope * xt + result.intercept + epsilon
    yt = regress[1::] - r * regress[0:-1]
    xt = xt[1::] - r * xt[0:-1]
    result1 = stats.linregress(xt, yt)
    epsilon = signal.detrend(yt)
    r = np.corrcoef(epsilon[1::], epsilon[0:-1])[0, 1]
    slope_se = result1.stderr
    intercept_se = result1.intercept_stderr / (1 - r)
    slope = result.slope
    intercept = result.intercept
    res = {'slope': slope, 'intercept': intercept, 'pvalue': result.pvalue, 'rvalue': result.rvalue,
           'stderr': slope_se, 'intercept_stderr': intercept_se}
    return dotdict(res)






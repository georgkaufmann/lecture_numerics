"""
Lecture: Numerical methods in Geosciences
Chapter 08: Optimisation
(c) Georg Kaufmann
"""

def chi2(o,p,oerr):
    """
    ----------------------------------------------
    Least-squares fitting criterion
    input:
    o[n]    - observations
    p[n]    - predictions data
    oerr[n] - uncertainty in observations
    output:
    chi2    - least-square fit
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    """
    import numpy as np
    chi2 = 0
    for i in np.range(len(o)):
        chi2 = chi2 + ((o[i]-p[i])/(oerr[i]))**2
    return chi2



def fit_linear(x,y,yerr):
    """
    ----------------------------------------------
    fit a straight line to a set of data points
    input:
    x[n]    - independent data
    y[n]    - dependent data
    yerr[n] - uncertainty in dependent data
    output:
    a,b     - intercept and slope of best-fit line
    r       - regression coefficient [-1,1]
    chi2    - least-square fit
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    """
    import numpy as np
    n = len(x)
    # define weights
    s = 0.
    sx = 0.
    sy = 0.
    sxx = 0.
    sxy = 0.
    for i in range(n):
        s   = s + 1 / yerr[i]**2
        sx  = sx + x[i] / yerr[i]**2
        sy  = sy + y[i] / yerr[i]**2
        sxx = sxx + x[i]**2 / yerr[i]**2
        sxy = sxy + x[i]*y[i] / yerr[i]**2
    delta = s*sxx - sx**2
    # determine coefficients and uncertainties
    a     = (sxx*sy-sx*sxy) / delta
    b     = (s*sxy-sx*sy) / delta
    siga  = sxx / delta
    sigb  = s / delta
    
    # regression and chi2 value
    r     = -sx/np.sqrt(s*sxx)
    chi2  = 0.
    for i in range(n):
        chi2  = chi2 + ((y[i]-a-b*x[i])**2 / yerr[i]**2)
    
    return a,b,r,chi2


def fit_linear_function(f,x,y,yerr,m=4):
    """
    ----------------------------------------------
    fit a set of function with linear coefficients
    to  a set of data points
    input:
    f       - external function used for predictions
    x[n]    - independent data
    y[n]    - dependent data
    yerr[n] - uncertainty in dependent data
    m=4     - number of arguments (default m=4)
    output:
    a[m]    - model coefficients as array
    chi2    - least-square fit
    needs:
    lin_lu_decompose
    lin_lu_solve
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    """
    import numpy as np
    import numerics.lingl
    n = len(x)
    alpha = np.zeros(m*m).reshape(m,m)
    beta  = np.zeros(m)
    # fill design matrix and rhs vector
    for j in range(m):
        for k in range(m):
            for i in range(n):    
                alpha[k,j] = alpha[k,j] + f(j,x[i])*f(k,x[i])/yerr[i]**2

    for k in range(m):
        for i in range(n):
            beta[k] = beta[k] + y[i]*f(k,x[i])/yerr[i]**2
    # solve system with LU decomposition
    l,u = numerics.lingl.lin_lu_decompose(alpha)
    a = numerics.lingl.lin_lu_solve(l,u,beta)
    # chi2 value
    chi2  = 0.
    for i in range(n):
        model = 0.
        for j in range(m):
            model = model + a[j]*f(j,x[i])
        chi2 = chi2 + ((y[i]-model)**2 / yerr[i]**2)
    return a,chi2

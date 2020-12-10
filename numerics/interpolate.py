"""
Lecture: Numerical methods in Geosciences
Chapter 04: Interpolation
(c) Georg Kaufmann
"""


def pol_lagrange_basis(xint,xdata,n,k):
    '''
    ----------------------------------------------
    calculates Lagrange basis function L_{n,k}
    input:
    xint     - x coordinate
    xdata[n] - data points
    n        - degree of Lagrange base function
    m        - order of Lagrange base function
    output:
    P        - Lagrange base function at point xint
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    P = 1
    for i in np.arange(0,n+1):
        if (i!=k):
            P = P * (xint-xdata[i]) / (xdata[k]-xdata[i])
    return P


def pol_lagrange (xint,xdata,ydata):
    '''
    ----------------------------------------------
    calculate Lagrange interpolation polynomial
    input:
    xint     - x coordinate
    xdata[0:n] - data points x value
    ydata[0:n] - data points y value
    output:
    yint  - value of Lagrange interpolation polynomial
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    n = len(xdata)-1
    yint = 0.
    for k in np.arange(0,n+1):
        yint = yint + ydata[k] * pol_lagrange_basis(xint,xdata,n,k)
    return yint


def pol_spline10(x,y):
    '''
    ----------------------------------------------
    calculation of coefficients for linear spline
    input:
    x[n]   - independent data coordinate
    y[n]   - dependent data coordinate
    output:
    b10    - array of slope coefficients
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    n = len(x)
    b10 = np.zeros(n)
    for i in np.arange(0,n-1):
        b10[i] = (y[i+1]-y[i]) / (x[i+1]-x[i])
    return b10


def pol_splint10(xint,x,y,b10):
    '''
    ----------------------------------------------
    calculation of linear spline
    input:
    xint   - interpolation point
    x[n]   - independent data coordinate
    y[n]   - dependent data coordinate
    b10    - array of slope coefficients
    output:
    yint10 - spline interpolation
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    n = len(x)
    # if xint is outside the x() interval take a boundary value (left or right)
    if (xint <= x[0]):
        yint10 = y[0]
    elif (xint >= x[n-1]):
        yint10 = y[n-1]
    # find interval and evaluate spline interpolation
    else:
        for i in np.arange(0,n-1):
            if (xint >= x[i] and xint <= x[i+1]):
                h      = xint - x[i]
                yint10 = y[i] + b10[i]*h
    return yint10


def pol_spline32(x,y):
    '''
    ----------------------------------------------
    calculation of coefficients for cubic spline
    input:
    x[n]   - independent data coordinate
    y[n]   - dependent data coordinate
    output:
    b32    - array of linear coefficients
    c32    - array of quadratic coefficients
    d32    - array of cubic coefficients
    needs:
    -
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    n = len(x)
    b32 = np.zeros(n)
    c32 = np.zeros(n)
    d32 = np.zeros(n)
    # step 1: preparation
    d32[0] = x[1] - x[0]
    c32[1] = (y[1] - y[0])/d32[0]
    for i in np.arange(2,n): # 2,n-1
        d32[i-1]   = x[i] - x[i-1]
        b32[i-1]   = 2.0*(d32[i-2] + d32[i-1])
        c32[i] = (y[i] - y[i-1])/d32[i-1]
        c32[i-1]   = c32[i] - c32[i-1]
    # step 2: end conditions
    b32[0]   = -d32[0]
    b32[n-1] = -d32[n-2]
    c32[0]   = 0.0
    c32[n-1] = 0.0
    if (n != 3):
        c32[0]   = c32[2]/(x[3]-x[1]) - c32[1]/(x[2]-x[0])
        c32[n-1] = c32[n-2]/(x[n-1]-x[n-3]) - c32[n-3]/(x[n-2]-x[n-4])
        c32[0]   = c32[0]*d32[0]**2/(x[3]-x[0])
        c32[n-1] = -c32[n-1]*d32[n-2]**2/(x[n-1]-x[n-4])
    # step 3: forward elimination
    for i in np.arange(2,n+1): # 2,n
         h        = d32[i-2]/b32[i-2]
         b32[i-1] = b32[i-1] - h*d32[i-2]
         c32[i-1] = c32[i-1] - h*c32[i-2]
    # step 4: back substitution
    c32[n-1] = c32[n-1]/b32[n-1]
    for j in np.arange(1,n): # 1,n-1
         i      = n-j
         c32[i-1] = (c32[i-1] - d32[i-1]*c32[i])/b32[i-1]
    # step 5: compute spline coefficients
    b32[n-1] = (y[n-1] - y[n-2])/d32[n-2] + d32[n-2]*(c32[n-2] + 2.0*c32[n-1])
    for i in np.arange(1,n): # i = 1,n-1
        b32[i-1] = (y[i] - y[i-1])/d32[i-1] - d32[i-1]*(c32[i] + 2.0*c32[i-1])
        d32[i-1] = (c32[i] - c32[i-1])/d32[i-1]
        c32[i-1] = 3.0*c32[i-1]
    c32[n-1] = 3.0*c32[n-1]
    d32[n-1] = d32[n-2]
    return b32,c32,d32


def pol_splint32(xint,x,y,b32,c32,d32):
    '''
    #----------------------------------------------------------------------
    calculation of cubic spline
    input:
    xint   - interpolation point
    x[n]   - independent data coordinate
    y[n]   - dependent data coordinate
    b32    - array of linear coefficients
    c32    - array of quadratic coefficients
    d32    - array of cubic coefficients
    output:
    yint32 - spline interpolation
    needs:
    -
    from: Lecture Numerical methods in geoscience
    #----------------------------------------------------------------------
    '''
    import numpy as np
    n = len(x)
    # if xint is outside the x() interval take a boundary value (left)
    if (xint <= x[0]):
        yint32 = y[0]
    # if xint is outside the x() interval take a boundary value (right)
    elif (xint >= x[n-1]):
        yint32 = y[n-1]
    # find interval and evaluate spline interpolation
    else:
        for i in np.arange(0,n-1): # 1,n-1
            if (xint >= x[i] and xint <= x[i+1]):
                h      = xint - x[i]
                yint32 = y[i] + b32[i]*h + c32[i]*(h**2) + d32[i]*(h**3)
    return yint32

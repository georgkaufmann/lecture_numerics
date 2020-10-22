# import libraries
import numpy as np

def pol_lagrange (xint,xdata,ydata):
    '''
    #----------------------------------------------------------------------
    # given the arrays xdata(0:n) and ydata(0:n) of length n, which tabulate
    # a function (with the xdata's in ascending order), and a coordinate xint
    # calculate the Lagrange interpolation polynomial of order n
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    n = len(xdata)-1
    yint = 0.
    for k in np.arange(0,n+1):
        yint = yint + ydata[k] * pol_lagrange_basis(xint,xdata,n,k)
    return yint

def pol_lagrange_basis(xint,xdata,n,k):
    '''
    #----------------------------------------------------------------------
    # calculate Lagrange basis function L_{n,k}
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    P = 1
    for i in np.arange(0,n+1):
        if (i!=k):
            P = P * (xint-xdata[i]) / (xdata[k]-xdata[i])
    return P

def pol_spline10(x,y):
    '''
    #----------------------------------------------------------------------
    # subroutine calculates coefficients for linear spline
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    n = len(x)
    b10 = np.zeros(n)
    for i in np.arange(0,n-1):
        b10[i] = (y[i+1]-y[i]) / (x[i+1]-x[i])
    return b10


def pol_splint10(xint,x,y,b10):
    '''
    #----------------------------------------------------------------------
    # subroutine calculates linear spline
    # (c) Georg Kaufmann
    ----------------------------------------------------------------------
    '''
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
    # subroutine calculates coefficients for cubic spline
    # (c) Georg Kaufmann
    '''
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
    # subroutine calculates cubic spline
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
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



def pol_f(x):
    '''
    #----------------------------------------------------------------------
    # Calculate function
    #----------------------------------------------------------------------
    '''
    pol_f = x*np.exp(-x/5.e0)
    return pol_f


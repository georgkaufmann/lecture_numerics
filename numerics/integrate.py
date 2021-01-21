"""
Lecture: Numerical methods in Geosciences
Chapter 06: Integration
(c) Georg Kaufmann
"""

def simple_midpoint(f,a,b):
    """
    Simple midpoint integration
    """
    simple_midpoint = (b-a) * f((a+b)/2)
    return simple_midpoint


def simple_trapez(f,a,b):
    """
    Simple trapezoidal integration
    """
    simple_trapez = (b-a)/2. * (f(a) + f(b))
    return simple_trapez


def simple_simpson(f,a,b):
    """
    Simple Simpson integration
    """
    simple_simpson = (b-a)/6. * (f(a) + 4.*f((a+b)/2.)+ f(b))
    return simple_simpson


def int_trapez (int_f,a,b,n):
    '''
    #----------------------------------------------------------------------
    # function integrates the function f(x) between a and b
    # using the extended Trapezoidal rule for n subintervals
    # input:
    # a        - lower integration limit
    # b        - upper integration limit
    # n        - number of sub-intervals
    # f        - external function
    # output:
    # int_trapez - value of interval
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    import numpy as npY
    # calculate stepsize
    h = (b-a) / float(n)
    # calculate integral
    int_trapez = int_f(a) + int_f(b)
    for i in np.arange(1,n): # i = 1,n-1
        int_trapez = int_trapez + 2.0*int_f(a+i*h)
    int_trapez = h / 2.0 * int_trapez
    return int_trapez


def int_simpson (int_f,a,b,n):
    '''
    #----------------------------------------------------------------------
    # function integrates the function f(x) between a and b
    # using the extended Simpson rule for n subintervals
    # input:
    # a        - lower integration limit
    # b        - upper integration limit
    # n        - number of sub-intervals
    # f        - external function
    # output:
    # int_simpson - value of interval
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    import numpy as npY
    # calculate stepsize
    if (n%2 != 0):
        exit ('int_simpson: n must be even')
    h = (b-a) / float(n)
    int_simpson = int_f(a) + int_f(b)
    for i in np.arange(1,n/2):  # i = 1,n/2-1
        int_simpson = int_simpson + 2.0*int_f(a+(2*i)*h)
    for i in np.arange(1,n/2+1): #  1,n/2
        int_simpson = int_simpson + 4.0*int_f(a+(2*i-1)*h)
    int_simpson = h/3.0 * int_simpson
    return int_simpson

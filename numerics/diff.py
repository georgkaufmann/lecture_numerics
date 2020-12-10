"""
Lecture: Numerical methods in Geosciences
Chapter 05: Differentiation
(c) Georg Kaufmann
"""


def diff_forward2 (f,x,h):
    '''
    ----------------------------------------------
    approximate the first derivative of f(x), using 
    the two-point forward (or backward) method
    input:
    f        - external function
    x        - evalution point
    h        - stepsize
    f        - external function
    output:
    df   - value of derivative
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    df = (f(x+h) - f(x)) / h
    return df


def diff_central3 (f,x,h):
    '''
    ----------------------------------------------
    approximate the first derivative of f(x), using
    the three-point central-point method
    input:
    f        - external function
    x        - evalution point
    h        - stepsize
    f        - external function
    output:
    df   - value of derivative
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    df = (f(x+h) - f(x-h)) / (2*h)
    return df


def diff_central5 (f,x,h):
    '''
    ----------------------------------------------
    approximate the first derivative of f(x), using
    the five-point central-point method
    input:
    f        - external function
    x        - evalution point
    h        - stepsize
    f        - external function
    output:
    df   - value of derivative
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    df = (f(x-2*h) - 8.*f(x-h) + 8.*f(x+h) - f(x+2*h)) / (12*h)
    return df

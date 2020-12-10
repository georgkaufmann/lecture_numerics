"""
Lecture: Numerical methods in Geosciences
Chapter 03: Roots
(c) Georg Kaufmann
"""


def root_bracket (f,a,b,n):
    '''
    ----------------------------------------------
    Bracket possible roots of a function f in the
    interval x in [a,b] by dividing the
    interval in n sub-intervals and searching
    for sign changes of the function
    input:
    f       - function
    a       - left boundary
    b       - right boundary
    n       - number of sub-intervals
    output:
    nb      - number of intervals with roots
    xb1[nb] - left interval boundaries
    xb1[nb] - right interval boundaries
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    nb  = 0
    xb1 = np.zeros([0])
    xb2 = np.zeros([0])
    x   = a
    dx  = (b-a)/n
    fa  = f(x)
    for i in np.linspace(1,n,n,dtype=int):
        x = x + dx
        fb = f(x)
        if (fa*fb <= 0):
            nb = nb + 1
            xb1 = np.append(xb1,x-dx)
            xb2 = np.append(xb2,x)
        fa = fb
    return xb1,xb2,nb 


def root_bisection (f,a,b,tol):
    '''
    ----------------------------------------------
    Find root of a function f in the interval [a,b]
    with the bisection method
    The interval should contain a sign change
    Use root_bracket() before ...
    input:
    f       - function
    a       - left boundary of bracketed interval
    b       - right boundary of bracketed interval
    tol     - accuracy for root
    output:
    root_bisection - x coordinate of root
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    nmax = 20
    fa = f(a)
    fb = f(b)
    # exit, when root is not bracketed
    if (fa*fb > 0):
        print ('root_bisection: root not bracketed')
        exit()
    # orient search such that f>0 lies at x+dx
    if (fa <= 0.):
        root_bisection = a
        dx            = b-a
    else:
        root_bisection = b
        dx            = a-b
    # find root, loop over nmax, and exit, if f=0 or dx<tol
    for i in np.arange(1,nmax):
        a = root_bisection
        b = root_bisection+dx
        dx   = dx*0.5
        x0   = root_bisection + dx
        fx0  = f(x0)
       #print (i,a,b,dx,x0,fx0)
        if (fx0 <= 0.):
            root_bisection = x0
        if (np.abs(dx) <= tol or fx0 == 0.):
            return root_bisection
    exit ('root_bisection: too many iterations in root_bisection')


def root_secant(f,a,b,tol):
    '''
    ----------------------------------------------
    Find root of a function f in the interval [a,b]
    with the secant method
    The interval should contain a sign change
    Use root_bracket() before ...
    input:
    f       - function
    a       - left boundary of bracketed interval
    b       - right boundary of bracketed interval
    tol     - accuracy for root
    output:
    root_secant - x coordinate of root
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    nmax = 20
    for i in np.arange(1,nmax):
        fa   = f(a)
        fb   = f(b)
        x0   = b - fb * (b-a) / (fb-fa)
        fx0  = f(x0)
        dx   = b-a
       #print (i,a,b,dx,x0,fx0)
        a  = b
        b  = x0
        root_secant = x0
        if (np.abs(dx) <= tol or fx0 == 0.):
            return root_secant
    exit ('root_secant: too many iterations in root_secant')



def root_newton (f,df,a,tol):
    '''
    ----------------------------------------------
    Find root of a function f close to point a
    with the Newton-Raphson method
    input:
    a           - initial gues of root
    f,df        - function and its first derivative
    acc         - desired tolerance 
    output:
    root_newton - root
    needs:
    -   
    from: Lecture Numerical methods in geoscience
    ----------------------------------------------
    '''
    import numpy as np
    nmax = 20
    root_newton = a
    for i in np.arange(1,nmax):
        root_newton   = root_newton - f(root_newton) / df(root_newton)
        fa = f(root_newton)
       #print (i,a,0.e0,0.e0,root_newton,fa)
        if (np.abs(fa) <= tol):
            return root_newton
    exit ('root_newton: too many iterations in root_newton')

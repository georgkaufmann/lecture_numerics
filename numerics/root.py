# import libraries
import numpy as np

def root_bracket (a,b,n):
    '''
    #-----------------------------------------------------------------------
    # Given a function fx defined on the intervall [a,b], subdivide the
    # intervall in n equally spaced segments, and search for zero crossings
    # of the function. nb is input as the maximum number of roots searched,
    # it is reset to the number of bracketing pairs found, xb1(1:nb),xb2(1:nb).
    # (c) Numerical recipes 
    #-----------------------------------------------------------------------
    '''
    nb  = 0
    xb1 = np.zeros([0,0])
    xb2 = np.zeros([0,0])
    x   = a
    dx  = (b-a)/n
    fa  = root_f(x)
    for i in np.linspace(1,n,n,dtype=int):
        x = x + dx
        fb = root_f(x)
        if (fa*fb <= 0):
            nb = nb + 1
            xb1 = np.append(xb1,x-dx)
            xb2 = np.append(xb2,x)
        fa = fb
    return xb1,xb2,nb


def root_bisection (a,b,tol):
    '''
    #-----------------------------------------------------------------------
    # Given a function fx defined on the interval [a,b], which contains
    # a possible root (bracketing!), the root is found by halving the
    # interval, until the desired accuracy +/-acc is achieved, otherwise
    # the algorithm quits.
    # (c) modified from Numerical recipes
    #-----------------------------------------------------------------------
    '''
    nmax = 20
    fa = root_f(a)
    fb = root_f(b)
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
        fx0  = root_f(x0)
       #print (i,a,b,dx,x0,fx0)
        if (fx0 <= 0.):
            root_bisection = x0
        if (np.abs(dx) <= tol or fx0 == 0.):
            return root_bisection
    exit ('root_bisection: too many iterations in root_bisection')


def root_secant(a,b,tol):
    '''
    #-----------------------------------------------------------------------
    # Given a function fx defined on the intervall [a,b], which contains
    # a possible root (bracketing!), the root is found by halving the
    # interval, until the desired accuracy +/-acc is achieved, otherwise
    # the algorithm quits.
    # (c) Georg Kaufmann
    #-----------------------------------------------------------------------
    '''
    nmax = 20
    for i in np.arange(1,nmax):
        fa   = root_f(a)
        fb   = root_f(b)
        x0   = b - fb * (b-a) / (fb-fa)
        fx0  = root_f(x0)
        dx   = b-a
       #print (i,a,b,dx,x0,fx0)
        a  = b
        b  = x0
        root_secant = x0
        if (np.abs(dx) <= tol or fx0 == 0.):
            return root_secant
    exit ('root_secant: too many iterations in root_secant')


def root_newton (a,tol):
    '''
    #-----------------------------------------------------------------------
    # Given a function fx defined on the interval [a,b], which contains
    # a possible root (bracketing!), the root is found the Newton method.
    # Input:
    # a           - initial gues of root
    # f,df        - function and its first derivative
    # acc         - desired tolerance 
    #Output:
    # root_newton - root
    # interval, until the desired accuracy +/-acc is achieved, otherwise
    # the algorithm quits.
    # (c) modified from Numerical recipes 
    #-----------------------------------------------------------------------
    '''
    nmax = 20
    root_newton = a
    for i in np.arange(1,nmax):
        root_newton   = root_newton - root_f(root_newton) / root_df(root_newton)
        fa = root_f(root_newton)
       #print (i,a,0.e0,0.e0,root_newton,fa)
        if (np.abs(fa) <= tol):
            return root_newton
    exit ('root_newton: too many iterations in root_newton')




def root_f(x):
    '''
    #-----------------------------------------------------------------------
    # provide function f(x)
    #-----------------------------------------------------------------------
    '''
    root_f = np.sin(x)
    return root_f

def root_df(x):
    '''
    #-----------------------------------------------------------------------
    # provide first derivative of function f(x)
    #-----------------------------------------------------------------------
    '''
    root_df = np.cos(x)
    return root_df

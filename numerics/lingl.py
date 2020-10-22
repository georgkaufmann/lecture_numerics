# import libraries
import numpy as np

def lin_gauss(a,b,n):
    '''
    #----------------------------------------------------------------------
    # subroutine solves the system of linear equations
    # a(n,n)*x(n) = b(n)
    # using the Gauss algorithm with pivoting
    # Input:
    # n       - number of equations
    # a(n,n)  - coefficient matrix
    # b(n)    - rhs vector
    # Output:
    # x(n)    - solution vector
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    # Gauss elimination with pivoting
    for i in np.arange(0,n): # 1,n-1
        for k in np.arange(i+1,n): # i+1,n
            if (a[i][i] == 0.):
                exit ('lin_gauss: pivot element is zero')
            pivot = a[k][i] / a[i][i]
            for j in np.arange(0,i):
                a[k,j] = 0.0     # not really needed
            for j in np.arange(i+1,n): # i+1,n
                a[k][j] = a[k][j] - pivot*a[i][j]
            b[k] = b[k] - pivot*b[i]
    # solve reduced system with backward substitution
    x = np.zeros([n])
    for i in np.arange(n-1,-1,step=-1): # n,1,-1
        sum = 0.0
        for j in np.arange(i+1,n): # i+1,n
            sum = sum + a[i][j] * x[j]
        x[i] = (b[i] - sum) / a[i][i]
    return x


def lin_lu(a,b,n):
    '''
    #-----------------------------------------------------------------------
    # subroutine solves the system of linear equations
    # a(n,n)*x(n) = b(n)
    # using the LU decomposition
    # Input:
    # n       - number of equations
    # a(n,n)  - coefficient matrix
    # b(n)    - rhs vector
    # Output:
    # x(n)    - solution vector
    # (c) Georg Kaufmann
    #-----------------------------------------------------------------------
    '''
    # use LU decomposition Ax=LUx=b => Ly=b and Ux=y
    l = np.zeros([n,n])
    u = np.zeros([n,n])
    for j in np.arange(1,n+1): # 1,n
        l[j-1][j-1] = 1.
        u[1-1][j-1] = a[1-1][j-1]
        l[j-1][1-1] = a[j-1][1-1] / u[1-1][1-1]
    for i in np.arange(2,n+1): # 2,n
        sum = 0.
        for k in np.arange(1,i): # 1,i-1
            sum = sum + l[i-1][k-1]*u[k-1][i-1]
        u[i-1][i-1] = a[i-1][i-1] - sum
        for j in np.arange(i+1,n+1): # i+1,n
            sum = 0.
            for k in np.arange(1,i): # 1,i-1
                sum = sum + l[i-1][k-1]*u[k-1][j-1]
            u[i-1][j-1] = (a[i-1][j-1] -  sum) / l[i-1][i-1]
            sum = 0.
            for k in np.arange(1,i): # 1,i-1
                sum = sum + l[j-1][k-1]*u[k-1][i-1]
            l[j-1][i-1] = (a[j-1][i-1] - sum) / u[i-1][i-1]
    # solve decomposed system Ly=b with forward substitution
    for i in np.arange(1,n+1): # 1,n
        sum = 0.
        for j in np.arange(1,i): # 1,i-1
            sum = sum + l[i-1][j-1] * b[j-1]
        b[i-1] = (b[i-1] - sum) / l[i-1][i-1]
    # solve decomposed system Ux=y with backward substitution
    for i in np.arange(n,0,step=-1): # n,1,-1
        sum = 0.
        for j in np.arange(i+1,n+1): # i+1,n
            sum = sum + u[i-1][j-1] * b[j-1]
        b[i-1] = (b[i-1] - sum) / u[i-1][i-1]
    return b


def lin_f(x):
    '''
    #----------------------------------------------------------------------
    # provide function f(x)
    #----------------------------------------------------------------------
    '''
    lin_f = x*np.exp(x)
    return lin_f


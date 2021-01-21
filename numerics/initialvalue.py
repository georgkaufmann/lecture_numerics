"""
Lecture: Numerical methods in Geosciences
Chapter 09: Initial-value problems
(c) Georg Kaufmann
"""

# import libraries
import numpy as np

def ivp_euler(rhs,t,y,h):
    '''
    #----------------------------------------------------------------------
    # subroutine performs simple Euler integration for ordinary DGL
    # y' = f(t,y), y(0)=y_1, t in [a,b]
    # Input:
    # rhs   - right hand side of differential equation
    # t     - integration variable t_i
    # y     - approximate solution w_i
    # h     - step size
    # Output:
    # ynew  - new approximate solution w_(i+1)
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    k1 = rhs(t,y) 
    ynew = y + h * k1
    return ynew


def ivp_eulermod(rhs,t,y,h):
    '''
    #----------------------------------------------------------------------
    # subroutine performs modified Euler integration for ordinary DGL
    # y' = f(t,y), y(0)=y_1, t in [a,b]
    # Input:
    # rhs   - right hand side of differential equation
    # t     - integration variable t_i
    # y     - approximate solution w_i
    # h     - step size
    # Output:
    # ynew  - new approximate solution w_(i+1)
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    k1   = rhs(t,y)
    k2   = rhs(t+h,y+h*k1)
    ynew = y + h / 2. * (k1+k2)
    return ynew


def ivp_rk4(rhs,t,y,h):
    '''
    #----------------------------------------------------------------------
    # subroutine performs Runge-Kutta fourth-order integration for ordinary DGL
    # y' = f(t,y), y(0)=y_1, t in [a,b]
    # Input:
    # rhs   - right hand side of differential equation
    # t     - integration variable t_i
    # y     - approximate solution w_i
    # h     - step size
    # Output:
    # ynew  - new approximate solution w_(i+1)
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    k1   = h*rhs(t,y);
    k2   = h*rhs(t+h/2,y+k1/2);
    k3   = h*rhs(t+h/2,y+k2/2);
    k4   = h*rhs(t+h,y+k3);
    ynew = y + 1 / 6 * (k1+2*k2+2*k3+k4);
    return ynew


def ivp_adamsmoult3(rhs,t,y4,y3,y2,y1,h):
    '''
    #----------------------------------------------------------------------
    # subroutine performs Adam-Moulton three-step explicit integration
    # for ordinary DGL y' = f(t,y), y(0)=y_1, t in [a,b]
    # Input:
    # rhs   - right hand side of differential equation
    # t     - integration variable t_i
    # y1    - approximate solution w_(i+1)
    # y2    - approximate solution w_(i)
    # y3    - approximate solution w_(i-1)
    # y4    - approximate solution w_(i-2)
    # h     - step size
    # Output:
    # ynew  - new approximate solution w_(i+1)
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    k1   = rhs(t+h,y1)
    k2   = rhs(t,y2)
    k3   = rhs(t-h,y3)
    k4   = rhs(t-2*h,y4)
    ynew = y2 + h / 24. * (9.*k1 + 19.*k2 - 5.*k3 + k4)
    return ynew


def ivp_adamsbash4(rhs,t,y4,y3,y2,y1,h):
    """
    #-----------------------------------------------------------------------
    # subroutine performs Adam-Bashforth four-step explicit integration 
    # for ordinary DGL y' = f(t,y), y(0)=y_1, t in [a,b]
    # Input:
    # rhs   - right hand side of differential equation
    # t     - integration variable t_i
    # y1    - approximate solution w_(i)
    # y2    - approximate solution w_(i-1)
    # y3    - approximate solution w_(i-2)
    # y4    - approximate solution w_(i-3)
    # h     - step size
    # Output:
    # ynew  - new approximate solution w_(i+1)
    # (c) Georg Kaufmann
    #-----------------------------------------------------------------------
    """
    k1   = rhs(t,y1)
    k2   = rhs(t-h,y2)
    k3   = rhs(t-2*h,y3)
    k4   = rhs(t-3*h,y4)
    ynew = y1 + h / 24. * (55.*k1 - 59.*k2 + 37.*k3 - 9.*k4)
    return ynew


def ivp_rk4_fehlberg(rhs,a,b,alpha,hmin,hmax,tol):
    """
    !-----------------------------------------------------------------------
    ! subroutine performs Runge-Kutta Fehlberg integration for ordinary DGL
    ! with adaptable stepsize
    ! y' = f(t,y), y(0)=y_1, t in [a,b]
    ! Input:
    ! rhs   - right hand side of differential equation
    ! a,b   - start, end point of interval
    ! hmin  - minimum step size
    ! hmax  - maximum step size
    ! y     - approximate solution w_i
    ! tol   - accuracy
    ! Output:
    ! ynew  - new approximate solution w_(i+1)
    ! (c) Georg Kaufmann
    !-----------------------------------------------------------------------
    """
    h = hmax
    i = 0
    t = []; y = []
    t.append(a)
    y.append(alpha)
    while (t[i] < b):
        # coefficients for both Runge-Kutta methods
        k1 = h*rhs(t[i],y[i])
        k2 = h*rhs(t[i]+h/4,y[i]+k1/4)
        k3 = h*rhs(t[i]+3*h/8,y[i]+(3*k1+9*k2)/32)
        k4 = h*rhs(t[i]+12*h/13,y[i]+(1932*k1-7200*k2+7296*k3)/2197)
        k5 = h*rhs(t[i]+h,y[i]+439*k1/216-8*k2+3680*k3/513-845*k4/4104)
        k6 = h*rhs(t[i]+h/2,y[i]-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)
        # difference |w_i - tilde(w_i)| / h
        r  = np.abs(k1/360-128*k3/4275-2197*k4/75240+k5/50+2*k6/55)/h
        # check step size, avoid underflow
        if (r > 1.0e-20):
            delta = 0.84*(tol/r)**0.25
        else:
            delta = 10.
        # if tolerance is achieved, proceed
        if (r <= tol):
            i = i + 1
            t.append(t[i-1]+h)
            y.append(y[i-1] + 25*k1/216+1408*k3/2565+2197*k4/4104-k5/5)
        # estimate new step size
        if (delta <= 0.1):
            h = 0.1*h
        elif (delta >= 4.):
            h = 4.*h
        else:
            h = delta*h
        # check if stepsize will become too large or too small
        if (h > hmax):
            h = hmax
        if (h < hmin):
            print ('odei_rk_fehlberg: h < hmin')
        # check that step will not exceed upper limit
        if (t[i]+h > b):
            h = b-t[i]
    return t,y

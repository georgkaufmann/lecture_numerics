# import libraries
import numpy as np

def odei_euler(t,y,h):
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
    k1 = odei_f(t,y) 
    ynew = y + h * k1
    return ynew


def odei_eulermod(t,y,h):
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
    k1   = odei_f(t,y)
    k2   = odei_f(t+h,y+h*k1)
    ynew = y + h / 2. * (k1+k2)
    return ynew


def odei_rk4(t,y,h):
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
    k1   = h*odei_f(t,y);
    k2   = h*odei_f(t+h/2,y+k1/2);
    k3   = h*odei_f(t+h/2,y+k2/2);
    k4   = h*odei_f(t+h,y+k3);
    ynew = y + 1 / 6 * (k1+2*k2+2*k3+k4);
    return ynew


def odei_adamsmoult3(t,y4,y3,y2,y1,h):
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
    k1   = odei_f(t+h,y1)
    k2   = odei_f(t,y2)
    k3   = odei_f(t-h,y3)
    k4   = odei_f(t-2*h,y4)
    ynew = y2 + h / 24. * (9.*k1 + 19.*k2 - 5.*k3 + k4)
    return ynew


def odei_adamsbash4(t,y4,y3,y2,y1,h):
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
    k1   = odei_f(t,y1)
    k2   = odei_f(t-h,y2)
    k3   = odei_f(t-2*h,y3)
    k4   = odei_f(t-3*h,y4)
    ynew = y1 + h / 24. * (55.*k1 - 59.*k2 + 37.*k3 - 9.*k4)
    return ynew




def odei_f(t,y):
    '''
    #----------------------------------------------------------------------
    # Calculate weird function
    #----------------------------------------------------------------------
    '''
    odei_f = y - t**2 + 1
    return odei_f



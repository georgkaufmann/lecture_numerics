# import libraries
import numpy as np

def diff_forward2 (diff_f,x,h):
    '''
    #----------------------------------------------------------------------
    # function approximates the first derivative of f(x), using 
    # the two-point forward (or backward) method
    # input:
    # x        - evalution point
    # h        - stepsize
    # f        - external function
    # output:
    # df   - value of derivative
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    df = (diff_f(x+h) - diff_f(x)) / h
    return df

def diff_end3 (diff_f,x,h):
    '''
    #----------------------------------------------------------------------
    # function approximates the first derivative of f(x), using
    # the three-point end-point method
    # input:
    # x        - evalution point
    # h        - stepsize
    # f        - external function
    # output:
    # df   - value of derivative
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    df = (-3.0*diff_f(x) + 4*diff_f(x+h) - diff_f(x+2.*h)) / (2*h)
    return df

def diff_central3 (diff_f,x,h):
    '''
    #----------------------------------------------------------------------
    # function approximates the first derivative of f(x), using
    # the three-point central-point method
    # input:
    # x        - evalution point
    # h        - stepsize
    # f        - external function
    # output:
    # df   - value of derivative
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    df = (diff_f(x+h) - diff_f(x-h)) / (2*h)
    return df

def diff_central5 (diff_f,x,h):
    '''
    #----------------------------------------------------------------------
    # function approximates the first derivative of f(x), using
    # the five-point central-point method
    # input:
    # x        - evalution point
    # h        - stepsize
    # f        - external function
    # output:
    # df   - value of derivative
    # (c) Georg Kaufmann
    #----------------------------------------------------------------------
    '''
    df = (diff_f(x-2*h) - 8.*diff_f(x-h) + 8.*diff_f(x+h) - diff_f(x+2*h)) / (12*h)
    return df




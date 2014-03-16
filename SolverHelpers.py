# -*- coding: utf-8 -*-
"""
Created on Thu Jan 02 10:29:13 2014

@author: Krszysztof Sopyła
@email: krzysztofsopyla@gmail.com
@license: MIT
"""


import numpy as np
import scipy.sparse as sp
from numba import autojit      


class Model:
    """Model class """
    
    SV_idx = []
    
    NSV = 0
    SV = []
    Classes = 0
    Classes_idx=(0,0)

    Obj = 0
    Iter = -1
    Rho=0

    Alpha = []
 
    def __init__(self):
        pass
        


@autojit
def Compute_Rho_numba(G,alpha,y,C):
    '''
    Compute rho with numba
    G - array like, gradient
    alpha - array like, alpha coef
    y - array like, labels
    C - scalar, penalty SVM pram
    '''
    ub=100000000
    lb=-10000000
    nr_free=0
    sum_free=0.0
    
    for i in xrange(0,G.shape[0]):
        yG = y[i]*G[i]
        ai = alpha[i]
        if(ai==0):
            if(y[i]>0):
                ub=min(ub,yG)
            else:
                lb=max(lb,yG)
        elif(ai==C):
            if(y[i]<0):
                ub=min(ub,yG)
            else:
                lb=max(lb,yG)
        else:
            nr_free+=1
            sum_free+=yG
    r=0        
    if(nr_free>0):
        r=sum_free/nr_free
    else:
        r=(ub+lb)/2
    
    return r
            
        
        
        

@autojit
def Update_gradient_numba(G,Ki,Kj,delta_i,delta_j):
    '''
    Updates gradients based on kernel i,j kolumns and deltas 
    Use numba for speed, much faster then vecotrization with numpy
    Parameters
    ------------
    G - array like, gradient for update
    Ki,Kj - array like,  i,j kernel kolumns
    delta_i,delta_j - scalras, delta alpha i and j
    '''
    for k in xrange(G.shape[0]):
        G[k]+=Ki[k]*delta_i+Kj[k]*delta_j            


@autojit
def Update_gradient_numba2Col(G,K2col,delta_i,delta_j):
    '''
    Updates gradients based on kernel i,j kolumns and deltas 
    Use numba for speed, much faster then vecotrization with numpy
    Parameters
    ------------
    G : array like, 
        gradient for update
    K2col : array like,  
        concatenated i,j kernel kolumns
    delta_i,delta_j : int
        delta alpha i and j
    '''
    
    n = G.shape[0]
    for k in xrange(n):
        G[k]+=K2col[k]*delta_i+K2col[n+k]*delta_j 
    

@autojit
def Update_gradient_numba_2col(G,K,delta_i,delta_j):
    '''
    Updates gradients based on kernel i,j kolumns and deltas 
    Use numba for speed, much faster then vecotrization with numpy
    Parameters
    ------------
    G: array like, 
        gradient for update
    K : array like,  
        array contains two concatenated i,j kernel kolumns
    delta_i,delta_j - float
        delta alpha i and j
    '''
    g_size = G.shape[0]
    for k in xrange(g_size):
        G[k]+=K[k]*delta_i+K[g_size+k]*delta_j            
        
    
        
@autojit
def FindMaxMinGrad(A,B,alpha,grad,y):
    '''
    Finds i,j indices with maximal violatin pair scheme
    A,B - 3 dim arrays, contains bounds A=[-C,0,0], B=[0,0,C]
    alpha - array like, contains alpha coeficients
    grad - array like, gradient
    y - array like, labels
    '''
    GMaxI=-100000
    GMaxJ=-100000
    
    GMax_idx=-1
    GMin_idx=-1
    
    for i in range(0,alpha.shape[0]):
        
        if (y[i] * alpha[i]< B[y[i]+1]):
            if( -y[i]*grad[i]>GMaxI):
                GMaxI= -y[i]*grad[i]
                GMax_idx = i
                
        if (y[i] * alpha[i]> A[y[i]+1]):
            if( y[i]*grad[i]>GMaxJ):
                GMaxJ= y[i]*grad[i]
                GMin_idx = i
                
    return (GMaxI,GMaxJ,GMax_idx,GMin_idx)                
                
      
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 02 13:38:09 2014

@author: Krszysztof Sopyła
@email: krzysztofsopyla@gmail.com
@license: MIT
"""


import numpy as np
import scipy.sparse as sp
import pylru


class Kernel:
    """Base kernel class """
    
    cache_size =100
   
    def __init__(self,cache_size=100):
        self.cache_size=cache_size
        
        
        
    def init(self,X,Y):
        
        #assert X.shape[0]==Y.shape[0]
        
        self.N = X.shape[0]
        column_size = self.N*4
        cacheMB = self.cache_size*1024*1024 #100MB for cache size   
        
        #how many kernel colums will be stored in cache
        cache_items = np.floor(cacheMB/column_size).astype(int)
        
        cache_items = min(self.N,cache_items)
        self.kernel_cache = pylru.lrucache(cache_items)        
        
        self.X =X
        self.Y = Y   
        
        self.compute_diag()
        
        
    def clean(self):
        """ clean the kernel cache """
        self.kernel_cache.clear()

class Linear(Kernel):
    """Linear Kernel"""
    
    def K(self,i):
        """ computes i-th kernel column """
        
        

        if( i in self.kernel_cache):
            Ki = self.kernel_cache[i]
        else:
            #for dense numpy array
            #ki = np.dot(self.X,vec)

            #for sparse matrix                                
            
            # first variant
#            vec = self.X[i,:];
#            Ki = self.X.dot(vec.T)
#            Ki = Ki.toarray()
            
            
            vec = self.X[i,:].toarray().flatten();
            Ki = self.X.dot(vec.T)
            
            
            #insert into cache
            self.kernel_cache[i]=Ki

        return Ki
    def K_vec(self,vec):
        
        return self.X.dot(vec.T)        
        
        
    def compute_diag(self):
        """
        Computes kernel matrix diagonal
        """
        # can be implementen in two ways
        #1. (x**2).sum(1) or
        #2. np.einsum('...i,...i',x,x)
        #second one is faster
        
        if(sp.issparse(self.X)):
            # result as matrix
            self.Diag = self.X.multiply(self.X).sum(1)
            #result as array
            self.Diag = np.asarray(self.Diag).flatten()
        else:
            self.Diag =np.einsum('...i,...i',self.X,self.X)
        


class RBF(Kernel):
    """RBF Kernel"""
    gamma=1.0
    
    
    def K(self,i):
        """ computes i-th kernel column """ 
        if( i in self.kernel_cache):
            Ki = self.kernel_cache[i]
        else:
            vec = self.X[i,:].toarray().flatten()           
            xi2=self.Xsquare[i]
            Ki =np.exp(-self.gamma*(xi2+ self.Xsquare-2*self.X.dot(vec.T)) ) 
            #insert into cache
            self.kernel_cache[i]=Ki

        return Ki
        
    def K_vec(self,vec):
        '''
        vec - array-like, row ordered data, should be not to big
        '''
        
        dot=self.X.dot(vec.T)  
        x2=self.Xsquare.reshape((self.Xsquare.shape[0],1))
        if(sp.issparse(vec)):        
            v2 = vec.multiply(vec).sum(1).reshape((1,vec.shape[0]))        
        else:
            v2 =  np.einsum('...i,...i',vec,vec)
        
        return np.exp(-self.gamma*(x2+v2-2*dot))
        
        
        
    def compute_diag(self):
        """
        Computes kernel matrix diagonal
        """
        
        #for rbf diagonal consists of ones exp(0)==1
        self.Diag = np.ones(self.X.shape[0])

        if(sp.issparse(self.X)):
            # result as matrix
            self.Xsquare = self.X.multiply(self.X).sum(1)
            #result as array
            self.Xsquare = np.asarray(self.Xsquare).flatten()
        else:
            self.Xsquare =np.einsum('...i,...i',self.X,self.X)
        
        
    


# -*- coding: utf-8 -*-
"""
Created on Sun Dec 01 19:50:27 2013

@author: ksirg
"""

import numpy as np
import scipy.sparse as sp
import pylru


                   
        
from numba import autojit      


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
                
                
                
#from numba import autojit
#import numba 
#
#numba.autojit
class Solver(object):
    """
    SVM solver class, 
    
    Uses first order SMO solver and CUDA for trainning acceleration
    """
    C=1;
    models=[]
    
    """original class labels """
    classes=[]
    
    nr_cls=2

    """ Data matrix """    
    X=[]
    
    """ Labels """
    Y=[]    
    
    
    
    _EPS = 0.001
    
    def __init__(self,X,Y,C=1):
        """ """
        
        
       
        self.C=C
        self.X = X
        self.Y =Y
        self.Y_map=Y

        self.N = np.shape(self.X)[0]       
        self.Dim = np.shape(self.X)[1]            
        
        self.order =np.zeros(self.N)
        self.count_cls=np.array([0])
        self.start_cls =np.array([0])
        self.start_cls=np.array([0])
    
           
        self.A=np.array( [-self.C,0,0])
        self.B=np.array( [0,0,self.C])
        
        #self.classes, self.idx_cls = np.unique(self.Y, return_inverse=True)
        cls, idx_cls = np.unique(self.Y, return_inverse=True)
        self.classes=cls
        self.idx_cls=idx_cls
        self.nr_cls = self.classes.shape[0]      
        
        self.new_classes = np.arange(0,self.nr_cls)
        
        
        
    def init(self, kernel):
        """ 
        Prepares data structures
        1. Group objects from particular class toogether
        2. Creates necessary data formats
        
        """
        
        self._group_classes()
          
        #kernel.init(self.X,self.Y)
        self.kernel = kernel
       

        
    def train(self):
        """
        Trains the svm
        """
      

        k=self.nr_cls
        #y=self.Y_map
        
        for i in range(k):
            #i_cls = self.new_classes[i]
            
            for j in range(i+1,k):
                
                #i - class start and end
                s_i,e_i=self.start_cls[i], self.start_cls[i+1]
                #j - class start and end
                (s_j,e_j)=(self.start_cls[j],self.start_cls[j+1])
                cls_i=(s_i,e_i)
                cls_j=(s_j,e_j)

                subProblem_idx = np.concatenate((np.arange(s_i,e_i),np.arange(s_j,e_j)) )
                #y_sub = y[subProblem_idx]
                #yi smaller then N
                #y_ij=np.concatenate( (y[s_i:e_i], y[s_j:e_j]))
                nn=self.count_cls[i]+self.count_cls[j]
                y_ij=np.concatenate( (-1*np.ones(self.count_cls[i],int), np.ones(self.count_cls[j],int) ) )
                
                size=y_ij.shape[0]
                #y_ij = y_ij.reshape(size,1)
                size=y_ij.shape[0]
                assert   nn==size             
                                
                alpha = np.zeros(size)
                
                #create and init new kernel
                sub_X = self.X[subProblem_idx,:]
                self.kernel.init(sub_X,y_ij)                
                
                model =self._solve(i,j,subProblem_idx,y_ij,alpha,self.kernel)  
                
                self.kernel.clean()
                
                
                self.models.append(model)
        
        
        
        
   
    def _solve(self,cls_i,cls_j,subProblem_idx,y_ij,alpha,kernel):
        """
        Solves dual L2-SVM for two classes

        Parameters
        -------------
        cls_i - tuple, contains numbers when 'i' class starst and ends in data, (cls_i[0],cls_i[1])
        cls_j - tuple, contains numbers when 'j' class starst and ends in data
        """
        
        n= y_ij.shape[0]
        #if alphas~=0 than gradient should be computed
        G = -1*np.ones(n)
        
        self._MAXITER=500000;
        
        #bug: is not the same as for whole dataset
        Kii= kernel.Diag
        
        iter=0;
        while(iter<self._MAXITER):
            #print iter,'\n-------------\n'
            
            (GMax_i,GMax_j,i,j)= self._select_working_set_numba(alpha,G,y_ij);
            if(GMax_i+GMax_j <self._EPS):
                break
           
            yi=y_ij[i]
            yj=y_ij[j]

            Ki= kernel.K(i)
            Ki= yi*y_ij*Ki
            
            Kj= kernel.K(j)
            Kj= yj*y_ij*Kj
           
            Kij = Ki[j]
           
           
            old_alpha_i = alpha[i]
            old_alpha_j = alpha[j]
 
            self._update_alpha(i,j,Kii[i],Kii[j],Kij,alpha,G,y_ij)
 
            delta_alpha_i = alpha[i] - old_alpha_i
            delta_alpha_j = alpha[j] - old_alpha_j        
           
            self._update_gradients(G,Ki,Kj,delta_alpha_i,delta_alpha_j)
            iter+=1
        
        
        
        #build model
        
        aNNZ = alpha.nonzero()
        
        sv_idx = subProblem_idx[aNNZ]

        obj=self._compute_obj(alpha,G)
        
        model = Model()
        model.SV_idx = sv_idx
        model.NSV = sv_idx.shape[0]
        model.SV = self.X[sv_idx,:]
        model.Classes = (self.classes[cls_i],self.classes[cls_j])
        model.Classes_idx=(cls_i,cls_j)
                
        model.Obj = obj
        model.Iter = iter
        model.Rho=self._compute_rho(G,alpha,y_ij)
        
        alpha=alpha*y_ij
        model.Alpha = alpha[aNNZ]
        
        return model
        
   
    def _compute_obj(self,alpha,G):
        nnzI = np.nonzero(alpha)
        objT= alpha[nnzI]*(G[nnzI]-1)
        obj = objT.sum()/2
            
#        obj=0.0
#        nS=alpha.shape[0]        
#        for nnzI in xrange(nS):
#            obj+= alpha[nnzI]*(G[nnzI]-1)            
#        obj/=2
        return obj
        
        
    def _compute_rho(self,G,alpha,y):
        '''
        computes rho for particular classifier
        '''
        C=self.C
        return Compute_Rho_numba(G,alpha,y,C)
    
    
    def _select_working_set(self,alpha,grad,y):
        """
        Function select variable to working set, finds pair of indices i,j
        such that
        i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        j: mimimizes -y_j * grad(f)_j, i in I_low(\alpha)
        
        
        'Support Vector Machine Solvers' - Leon Bottou, Chih-Jen Lin  
        
        Parameters
        ------------
        alpha   - alpha coeficients 
        grad    - gradient
        y       - labels
        
        Return
        -------------
        
        
        """
        A=self.A
        B=self.B
        C= self.C
        
        GMaxI=float('-inf')
        GMaxJ=float('-inf')
        
        GMax_idx=-1
        GMin_idx=-1
        
        n=alpha.shape[0]
        for i in xrange(0,n):
            
            yi=y[i]
            alpha_i=alpha[i]
            
            if (yi * alpha_i< B[yi+1]):
                if( -yi*grad[i]>GMaxI):
                    GMaxI= -yi*grad[i]
                    GMax_idx = i
                    
            if (yi * alpha_i> A[yi+1]):
                if( yi*grad[i]>GMaxJ):
                    GMaxJ= yi*grad[i]
                    GMin_idx = i
                            
        
        return (GMaxI,GMaxJ,GMax_idx,GMin_idx)
        
        
       
    def _select_working_set_numba(self,alpha,grad,y):
        """
        Function select variable to working set, finds pair of indices i,j
        such that
        i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        j: mimimizes -y_j * grad(f)_j, i in I_low(\alpha)
        
        
        'Support Vector Machine Solvers' - Leon Bottou, Chih-Jen Lin  
        
        Parameters
        ------------
        alpha   - alpha coeficients 
        grad    - gradient
        y       - labels
        
        Return
        -------------
        
        
        """
        A=self.A
        B=self.B
        #GMaxI,GMax_idx,z = test(grad)
        (GMaxI,GMaxJ,GMax_idx,GMin_idx)=FindMaxMinGrad(A,B,alpha,grad,y)                    
        
        return (GMaxI,GMaxJ,GMax_idx,GMin_idx)
    
    def _select_working_set_vec(self,alpha,grad,y):
        """
        Function select variable to working set, finds pair of indices i,j
        such that
        i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        j: mimimizes -y_j * grad(f)_j, i in I_low(\alpha)
        
        
        'Support Vector Machine Solvers' - Leon Bottou, Chih-Jen Lin  
        
        Use vectorization to speed up seeking proces of  i,j        
        
        
        Parameters
        ------------
        alpha   - alpha coeficients 
        grad    - gradient
        y       - labels
        
        Return
        -------------
        
        
        """
        C= self.C
        
        GMaxI=float('-inf')
        GMaxJ=float('-inf')
        
        GMax_idx=-1
        GMin_idx=-1
        
      
        filterB = (y*alpha < C*0.5*(y+1))
        values = -y * grad
        values[~filterB] =float('-inf')
        GMax_idx = values.argmax()
        GMaxI = -y[GMax_idx]*grad[GMax_idx]        
        
                
        filterA = y*alpha> -C*(0.5*(-y+1))
        values = y * grad
        values[~filterA] =float('-inf')
        GMin_idx = values.argmax()
        GMaxJ = y[GMin_idx]*grad[GMin_idx]
      
        return (GMaxI,GMaxJ,GMax_idx,GMin_idx)
    

        
    
    def _update_alpha(self,i,j,Kii,Kjj,Kij,alpha,G,y_ij):
        """
        updates alpha coeficients for particular problem
        Parameters
        -----------------
        i,j - indices
        Kii, Kjj - i-th and j-th kernel diagonal elements
        Kij - kernel value K(i,j)
        alpha - alpha coefictiens
        G - gradient
        y_ij - labels mapped to 1,-1, 0
        """
        yi=y_ij[i]
        yj=y_ij[j]
        quad_coef = Kii + Kjj - 2 * yi * yj * Kij;
        if (quad_coef <= 0):
            quad_coef = 1e-12

        C = self.C
        delta = 0;
        diff_alpha = 0;
        sum_alpha = 0;
       

        if (yi != yj):
            delta = (-G[i] - G[j]) / quad_coef
            diff_alpha = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta

            if (diff_alpha > 0):
                if (alpha[j] < 0):
                    alpha[j] = 0
                    alpha[i] = diff_alpha
            else:
                if (alpha[i] < 0):
                    alpha[i] = 0
                    alpha[j] = -diff_alpha


            if (diff_alpha > 0):
                if (alpha[i] > C):
                    alpha[i] = C
                    alpha[j] = C - diff_alpha
            else:
                if (alpha[j] > C):
                    alpha[j] = C
                    alpha[i] = C + diff_alpha
        else:
            delta = (G[i] - G[j]) / quad_coef
            sum_alpha = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta
            
            if (sum_alpha > C):
                if (alpha[i] > C):
                    alpha[i] = C
                    alpha[j] = sum_alpha - C
            else:
                if (alpha[j] < 0):
                    alpha[j] = 0
                    alpha[i] = sum_alpha
                    
            if (sum_alpha > C):
                if (alpha[j] > C):
                    alpha[j] = C
                    alpha[i] = sum_alpha - C
            else:
                if (alpha[i] < 0):
                    alpha[i] = 0
                    alpha[j] = sum_alpha
                    
                    
                    

    def _update_gradients(self,G,Ki,Kj,delta_i,delta_j):
        
        #n=G.shape[0]
        #for k in xrange(n):
        #G[k]+=Ki[k]*delta_i+Kj[k]*delta_j            
        #G+=delta_i*Ki+delta_j*Kj
        Update_gradient_numba(G,Ki,Kj,delta_i,delta_j)
    
    
    def _group_classes(self):
        
        y = self.Y
        x = self.X


        #self.classes, self.idx_cls = np.unique(y, return_inverse=True)
        #self.nr_cls = self.classes.shape[0]      
        #        
        #self.new_classes = np.arange(0,self.nr_cls)
        
        #contains mapped class [0,nr_cls-1]
        y_map = self.new_classes[self.idx_cls]
        
        #reorder the dataset, group class together
        order =np.argsort(a=y_map,kind='mergesort')
        
    
        
        self.X = x[order]
        self.Y = y[order]
        self.Y_map=y_map[order]
        self.order = order
        
        self.count_cls=np.bincount(y_map)
        
        self.start_cls = self.count_cls.cumsum()
        self.start_cls=np.insert(self.start_cls,0,0)
        
        
    def predict(self,X):
        ''' 
        Predicts the class
        
        Parameters
        -------------
        X - matrix with row vectors
        
        Returns
        ------------
        pred - array with class predictions
        dec_vals - array with class decision values
        '''

#        m=self.models[0]        
#        self.kernel.init(m.SV,[])
#        dec=self.kernel.K_vec(X)     
#        alpha = m.Alpha
#        #dec_vals=alpha.dot(dec_vals)
#        dec = dec.T.dot(alpha)
#        dec-=m.Rho
#        
#        dec_vals = dec
#        pred=-1*np.ones(dec.shape[0])
#        pred[dec_vals>0]=1
        
        

        mCount = len(self.models)
        nr_cls = self.nr_cls
        dataPoints = X.shape[0]
        dec_vals = np.zeros( (mCount,dataPoints) )        
        
        for i in xrange(mCount):
            m=self.models[i]        
            SV = m.SV
            self.kernel.init(SV,[])
            
            #big matrix nSV x nr_points (10k x 40k)
            
            #partX=X[]            
            dec=self.kernel.K_vec(X)
            
            alpha = m.Alpha
            #dec_vals=alpha.dot(dec_vals)
            dec = dec.T.dot(alpha)
            
            dec_vals[i,:]=dec-m.Rho
            #dec_vals[i,:]-=m.Rho
        
        
        votes=np.zeros((dataPoints,nr_cls))
                
        p=0
        for ci in xrange(nr_cls):
            for cj in xrange(ci+1,nr_cls):
                for d in xrange(dataPoints):
                    if(dec_vals[p,d]<0):
                        votes[d,ci]+=1
                    else:
                        votes[d,cj]+=1
                
                p+=1
        
        class_vote_max = np.argmax(votes,axis=1)
#        class_vote_max = np.zeros(dataPoints,dtype=int)
#        for c in xrange(nr_cls):
#            for d in xrange(dataPoints):
#                if(votes[d,c]> votes[d,class_vote_max[d]]):
#                    class_vote_max[d]=c
#                        
        #pred = np.zeros(dataPoints)
#        for d in xrange(dataPoints):
#            pred[d]=self.classes[class_vote_max[d]]
#            
        pred=self.classes[class_vote_max]
                
        return pred,dec_vals
           
      
      

class Model:
    """Model class """
    
    def __init__(self):
        pass
        

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
        v2 = vec.multiply(vec).sum(1).reshape((1,vec.shape[0]))        
        return np.exp(-self.gamma*(x2+v2-2*dot))
        
        
        
    def compute_diag(self):
        """
        Computes kernel matrix diagonal
        """
        self.Diag = np.ones(self.X.shape[0])

        if(sp.issparse(self.X)):
            # result as matrix
            self.Xsquare = self.X.multiply(self.X).sum(1)
            #result as array
            self.Xsquare = np.asarray(self.Xsquare).flatten()
        else:
            self.Xsquare =np.einsum('...i,...i',self.X,self.X)
        
        
    

















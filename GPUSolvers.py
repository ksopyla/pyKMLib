# -*- coding: utf-8 -*-
"""
Created on Sun Dec 01 19:50:27 2013

@author: ksirg
"""

import numpy as np
import itertools        

from SolverHelpers import *
#from kernels import *       

             
             
class GPUSVM2Col(object):
    """
    SVM solver class with cuda accelarated rbf kernel    
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
    
    def __init__(self,X,Y,C=1,concurrent_kernels=1,maxIter=500000):
        """ """    
        self._MAXITER=maxIter;
        self.C=C
        self.X = X.astype(np.float32)
        self.Y =Y
        self.Y_map=Y

        self.N, self.Dim = self.X.shape
        
        self.order =np.zeros(self.N)
        self.count_cls=np.array([0])
        self.start_cls =np.array([0])
        
        self.concurrent_kernels=concurrent_kernels
           
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
        
        
         #global kernel initialization            
        kernel.init_cuda(self.X,self.Y,self.start_cls,self.concurrent_kernels)        
        self.kernel = kernel
  
            
    def train(self):
        """
        Trains the svm
        """
      
        #all 2-element combination  
        #a=[x for x in itertools.combinations(np.arange(5),2)]
        #b=[ a[i:i+3] for i in range(0,len(a),3)]
        
        
        concurrent_kernels =self.concurrent_kernels
        
        #all combination, all class pair for partial classifier
        cls_pairs = [x for x in itertools.combinations(np.arange(self.nr_cls),2)]
        
        #class pairs grouped in concurrent_kernels chunks, which will be run in parallel
        cls_group = [ cls_pairs[i:i+concurrent_kernels] for i in range(0,len(cls_pairs),concurrent_kernels)]

        kernel = self.kernel
       
        
        for group in cls_group:           
            kernel_nr=0
            
            #init kernel for each class pair in group
            for pair in group:
                i=pair[0]
                j=pair[1]
                s_i,e_i=self.start_cls[i], self.start_cls[i+1]
                s_j,e_j=self.start_cls[j], self.start_cls[j+1]
                
                subProblem_idx = np.concatenate((np.arange(s_i,e_i),np.arange(s_j,e_j)) )


                count_i = self.count_cls[i]
                count_j = self.count_cls[j]
                
                y_cls=np.concatenate( (np.ones(count_i,dtype=np.int32), -1*np.ones(count_j,dtype=np.int32) ) )
              
               
                #allocate the memory for the labels and kernel output
                kernel.cls_init(kernel_nr,y_cls,i,j,count_i,count_j)
                

                sub_size=self.count_cls[i]+self.count_cls[j]
                alpha = np.zeros(sub_size)
                
                model =self._solve(i,j,subProblem_idx,y_cls,alpha,self.kernel,kernel_nr)  

                self.models.append(model)                
                
                self.kernel.clean(kernel_nr)

                kernel_nr+=1
                        
   
    def _solve(self,cls_1,cls_2,subProblem_idx,y_cls,alpha,kernel,kernel_nr):
        """
        Solves dual L2-SVM for two classes

        Parameters
        -------------
        cls_1 - int
            class 1 number
        cls_2 - int 
            class 2 number
        subProblem_idx : array-like
            contains sub problem indeces in whole dataset
        y_cls : array-like
            contains mapped class labelse (1 and -1)
        alpha: array-like
            alpha coeficients
        kernel: object
            GPUKernel
        kernel_nr: int
            concurrent kernel number
        """
        
        n= y_cls.shape[0]
        #if alphas~=0 than gradient should be computed
        G = -1*np.ones(n)
        
  
        
        
        Kii= kernel.Diag
        
        iter=0;
        K2col = np.zeros(2*n,dtype=np.float32)
        while(iter<self._MAXITER):
            #print iter,'\n-------------\n'
            
            (GMax_i,GMax_j,i,j)= self._select_working_set_numba(alpha,G,y_cls);
            
            #i,j = (0, 0)
            if(GMax_i+GMax_j <self._EPS):
                break
           
            if(i<self.count_cls[cls_1]):
                #i from class 1
                i_ds=i+self.start_cls[cls_1]
            else:
                #i from class 2
                i_ds=i-self.count_cls[cls_1]+self.start_cls[cls_2]
 

            if(j<self.count_cls[cls_1]):
                #j from class 1
                j_ds=j+self.start_cls[cls_1]
            else:
                #j from class 2
                j_ds=j-self.count_cls[cls_1]+self.start_cls[cls_2]

            
            K2col=kernel.K2Col(i,j,i_ds,j_ds,kernel_nr)
           
#            print '\n---------'
#            print 'iter=',iter
#            print 'i,j,i_ds,j_ds',i,j,i_ds,j_ds
#            print 'K2col'
#            print K2col[0:8]
#            print K2col[150:158]            
#            
#            print 'K2col>270'
#            print K2col[270:278]
#            print K2col[(270+150):(270+150+8)]            
            
            
            Kij = K2col[j]
           
           
            old_alpha_i = alpha[i]
            old_alpha_j = alpha[j]
 
            self._update_alpha(i,j,Kii[i_ds],Kii[j_ds],Kij,alpha,G,y_cls)
 
            delta_alpha_i = alpha[i] - old_alpha_i
            delta_alpha_j = alpha[j] - old_alpha_j        
           
            self._update_gradients(G,K2col,delta_alpha_i,delta_alpha_j)
            iter+=1
        
        #build model
        
        aNNZ = alpha.nonzero()
        
        sv_idx = subProblem_idx[aNNZ]

        obj=self._compute_obj(alpha,G)
        
        model = Model()
        model.SV_idx = sv_idx
        model.NSV = sv_idx.shape[0]
        model.SV = self.X[sv_idx,:]
        model.Classes = (self.classes[cls_1],self.classes[cls_2])
        model.Classes_idx=(cls_1,cls_2)
                
        model.Obj = obj
        model.Iter = iter
        model.Rho=self._compute_rho(G,alpha,y_cls)
        
        alpha=alpha*y_cls
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
                    
                    
                    

    def _update_gradients(self,G,K2col,delta_i,delta_j):
        
        #n=G.shape[0]
        #for k in xrange(n):
        #G[k]+=Ki[k]*delta_i+Kj[k]*delta_j            
        #G+=delta_i*Ki+delta_j*Kj
        Update_gradient_numba2Col(G,K2col,delta_i,delta_j)
    
    
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
        self.Y_map=y_map[order].astype(np.int32)
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
        
        partSize = 1000
        parts = int(np.ceil((0.0+dataPoints)/partSize))
        startPart=0
        endPart=partSize
        
        for i in xrange(mCount):
            m=self.models[i]        
            SV = m.SV
            self.kernel.predict_init(SV)
            alpha = m.Alpha
            for p in xrange(parts):
                startPart=p*partSize
                endPart=min(startPart+partSize,dataPoints)
                partX=X[startPart:endPart,:]  
                
                dec=self.kernel.K_vec(partX)
                dec = dec.T.dot(alpha)
                dec_vals[i,startPart:endPart]=dec-m.Rho
              
        
#        for i in xrange(mCount):
#            m=self.models[i]        
#            SV = m.SV
#            self.kernel.init(SV,[])
#            
#            #big matrix nSV x nr_points (10k x 40k)
#            dec=self.kernel.K_vec(X)
#            
#            alpha = m.Alpha
#            #dec_vals=alpha.dot(dec_vals)
#            dec = dec.T.dot(alpha)
#            
#            dec_vals[i,:]=dec-m.Rho
#            #dec_vals[i,:]-=m.Rho
        
        
        votes=np.zeros((dataPoints,nr_cls))
                
        p=0
        for ci in xrange(nr_cls):
            for cj in xrange(ci+1,nr_cls):
                for d in xrange(dataPoints):
                    if(dec_vals[p,d]>0):
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
           
      
      










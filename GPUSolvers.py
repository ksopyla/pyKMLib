# -*- coding: utf-8 -*-
"""
Created on Sun Dec 01 19:50:27 2013

@author: ksirg
"""



import numpy as np
import scipy.sparse as sp
import pylru


from solver_helpers import *
from kernels import *       
import itertools        

      
             
#from numba import autojit
#import numba 
#
#numba.autojit
             
             
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
           
      
      

import pycuda

import pycuda.driver as cuda
from pycuda.compiler import SourceModule        
import sparse_formats as spf

class GPURBF(object):
    """RBF Kernel with ellpack format"""
    
    cache_size =100
    
    Gamma=1.0
    
    #template
    func_name='rbfEllpackILPcol2multi'
    
    #template    
    module_file = 'KernelsEllpackCol2.cu'
    
    #template
    texref_nameI='VecI_TexRef'
    texref_nameJ='VecJ_TexRef'
    
    max_concurrent_kernels=1
   
    def __init__(self,gamma=1.0,cache_size=100):
        """
        Initialize object
        
        Parameters
        -------------
        
        max_kernel_nr: int
            determines maximal concurrent kernel column gpu computation
        """
        self.cache_size=cache_size
  
               
        self.threadsPerRow=1
        self.prefetch=2        
        
        self.tpb=128
        self.Gamma = gamma
       
        
        
        
        
    def init_cuda(self,X,Y, cls_start, max_kernels=1 ):
        
        #assert X.shape[0]==Y.shape[0]
        self.max_concurrent_kernels = max_kernels 
        
        self.X =X
        self.Y = Y
        
        self.cls_start=cls_start.astype(np.int32)
        
        #handle to gpu memory for y for each concurrent classifier
        self.g_y=[]
        #handle to gpu memory for results for each concurrent classifier
        self.g_out=[]
        self.kernel_out=[] 
        #blocks per grid for each concurrent classifier    
        self.bpg=[]
        
        #function reference
        self.func=[]
        
        #texture references for each concurrent kernel
        self.tex_ref=[]

        #main vectors 
        self.g_vecI=[]
        self.g_vecJ=[]
        
        self.main_vecI=[]
        self.main_vecJ=[]    
        
        self.cls_count=[]
        self.cls=[]
        self.g_cls_count=[]
        self.g_cls=[]
        
        self.sum_cls=[]
        
        for i in range(max_kernels):
            self.bpg.append(0)
            self.g_y.append(0)
            self.g_out.append(0)
            self.kernel_out.append(0)
            self.cls_count.append(0)
            self.cls.append(0)
            self.g_cls_count.append(0)
            self.g_cls.append(0)            
#            self.func.append(0)
#            self.tex_ref.append(0)
            self.g_vecI.append(0)
            self.g_vecJ.append(0)
#            self.main_vecI.append(0)
#            self.main_vecJ.append(0)
            self.sum_cls.append(0)
            
            
        self.N,self.Dim = X.shape
        column_size = self.N*4
        cacheMB = self.cache_size*1024*1024 #100MB for cache size   
        
        #how many kernel colums will be stored in cache
        cache_items = np.floor(cacheMB/column_size).astype(int)
        
        cache_items = min(self.N,cache_items)
        self.kernel_cache = pylru.lrucache(cache_items)        
        
        self.compute_diag()
           
        cuda.init()        
        
        self.dev = cuda.Device(0)
        self.ctx = self.dev.make_context()

        #reade cuda .cu file with module code        
        with open (self.module_file,"r") as CudaFile:
            module_code = CudaFile.read();
        
        #compile module
	#self.module = SourceModule(module_code,cache_dir='./nvcc_cache',keep=True,no_extern_c=True)
	self.module = SourceModule(module_code,keep=True,no_extern_c=True)
        
        
        (g_gamma,gsize)=self.module.get_global('GAMMA')       
        cuda.memcpy_htod(g_gamma, np.float32(self.Gamma) )
        
        #get functions reference

        Dim =self.Dim        
        vecBytes = Dim*4
        for f in range(self.max_concurrent_kernels):
            gfun = self.module.get_function(self.func_name)
            self.func.append(gfun)

            #init texture for vector I
            vecI_tex=self.module.get_texref('VecI_TexRef')
            self.g_vecI[f]=cuda.mem_alloc( vecBytes)           
            vecI_tex.set_address(self.g_vecI[f],vecBytes)

            #init texture for vector J
            vecJ_tex=self.module.get_texref('VecJ_TexRef')
            self.g_vecJ[f]=cuda.mem_alloc( vecBytes)     
            vecJ_tex.set_address(self.g_vecJ[f],vecBytes)
            
            self.tex_ref.append((vecI_tex,vecJ_tex) )
            
            self.main_vecI.append(np.zeros((1,Dim),dtype=np.float32))
            self.main_vecJ.append(np.zeros((1,Dim),dtype=np.float32))
            
            texReflist = list(self.tex_ref[f])
            
            #function definition P-pointer i-int
            gfun.prepare("PPPPPPiiiiiiPPP",texrefs=texReflist)
            
        
        #transform X to particular format
        v,c,r=spf.csr2ellpack(self.X,align=self.prefetch)
        #copy format data structure to gpu memory
        
        self.g_val = cuda.to_device(v)
        self.g_col = cuda.to_device(c)
        self.g_len = cuda.to_device(r)
        self.g_sdot = cuda.to_device(self.Xsquare)
        
        self.g_cls_start = cuda.to_device(self.cls_start)
        
        
        
        
    def cls_init(self,kernel_nr,y_cls,cls1,cls2,cls1_n,cls2_n):
        """
        Prepare cuda kernel call for kernel_nr, copy data for particular binary classifier, between class 1 vs 2.
         
        Parameters
        ------------
        kernel_nr : int
            concurrent kernel number
        y_cls : array-like
            binary class labels (1,-1)
        cls1: int
            first class number
        cls2: int
            second class number
        cls1_n : int
            number of elements of class 1
        cls2_n : int
            number of elements of class 2
        kernel_out : array-like
            array for gpu kernel result, size=2*len(y_cls)
        
        """
        warp=32
        align_cls1_n =  cls1_n+(warp-cls1_n%warp)%warp
        align_cls2_n =  cls2_n+(warp-cls2_n%warp)%warp
        
        self.cls1_N_aligned=align_cls1_n

        sum_cls= align_cls1_n+align_cls2_n   
        self.sum_cls[kernel_nr] = sum_cls
              
        
        self.cls_count[kernel_nr] = np.array([cls1_n,cls2_n],dtype=np.int32)
        self.cls[kernel_nr] = np.array([cls1,cls2],dtype=np.int32)  
        
        self.g_cls_count[kernel_nr] = cuda.to_device(self.cls_count[kernel_nr])
        
        self.g_cls[kernel_nr] = cuda.to_device(self.cls[kernel_nr])
        
        self.bpg[kernel_nr] =int( np.ceil( (self.threadsPerRow*sum_cls+0.0)/self.tpb ))
        
        self.g_y[kernel_nr] =  cuda.to_device(y_cls)
        
        self.kernel_out[kernel_nr] = np.zeros(2*y_cls.shape[0],dtype=np.float32)
        
        ker_out = self.kernel_out[kernel_nr]      
        self.g_out[kernel_nr] = cuda.to_device(ker_out) # cuda.mem_alloc_like(ker_out)
        
    
        #add prepare for device functions
        
    
    
    def K2Col(self,i,j,i_ds,j_ds,kernel_nr):
        """ 
        computes i-th and j-th kernel column 

        Parameters
        ---------------
        i: int
            i-th kernel column number in subproblem
        j: int
            j-th kernel column number in subproblem

        i_ds: int
            i-th kernel column number in whole dataset
        j_ds: int
            j-th kernel column number in  whole dataset

        kernel_nr : int
            number of concurrent kernel
            
        ker2ColOut: array like
            array for output
        
        Returns
        -------
        ker2Col
        
        """ 
        
        #make i-th and j-the main vectors
        vecI= self.main_vecI[kernel_nr]
        vecJ= self.main_vecJ[kernel_nr]
        
#        self.X[i_ds,:].todense(out=vecI)        
#        self.X[j_ds,:].todense(out=vecJ)  
        
        #vecI.fill(0)
        #vecJ.fill(0)
        
        
        
        #self.X[i_ds,:].toarray(out=vecI)        
        #self.X[j_ds,:].toarray(out=vecJ)        
        
        vecI=self.X.getrow(i_ds).todense()
        vecJ=self.X.getrow(j_ds).todense()
        
        
        #copy them to texture
        cuda.memcpy_htod(self.g_vecI[kernel_nr],vecI)
        cuda.memcpy_htod(self.g_vecJ[kernel_nr],vecJ)
        
#        temp = np.empty_like(vecI)
#        cuda.memcpy_dtoh(temp,self.g_vecI[kernel_nr])        
#        print 'temp',temp
        #lauch kernel
        
        gfunc=self.func[kernel_nr]
        gy = self.g_y[kernel_nr]
        gout = self.g_out[kernel_nr]
        gN = np.int32(self.N)
        g_i = np.int32(i)
        g_j = np.int32(j)
        g_ids = np.int32(i_ds)
        g_jds = np.int32(j_ds)
        gNalign = np.int32(self.cls1_N_aligned)
        gcs = self.g_cls_start
        gcc = self.g_cls_count[kernel_nr]
        gc  = self.g_cls[kernel_nr]
        bpg=self.bpg[kernel_nr]
        texReflist = list(self.tex_ref[kernel_nr])        
       
        #print 'start gpu i,j,kernel_nr ',i,j,kernel_nr
        #gfunc(self.g_val,self.g_col,self.g_len,self.g_sdot,gy,gout,gN,g_i,g_j,g_ids,g_jds,gNalign,gcs,gcc,gc,block=(self.tpb,1,1),grid=(bpg,1),texrefs=texReflist)
        #print 'end gpu',i,j
        #copy the results
       
        #grid=(bpg,1),block=(self.tpb,1,1)
        gfunc.prepared_call((bpg,1),(self.tpb,1,1),self.g_val,self.g_col,self.g_len,self.g_sdot,gy,gout,gN,g_i,g_j,g_ids,g_jds,gNalign,gcs,gcc,gc)
        
        cuda.memcpy_dtoh(self.kernel_out[kernel_nr],gout)

                
        
        return self.kernel_out[kernel_nr]
        
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
        
        return np.exp(-self.Gamma*(x2+v2-2*dot))
        
    def compute_diag(self):
        """
        Computes kernel matrix diagonal
        """
        
        #for rbf diagonal consists of ones exp(0)==1
        self.Diag = np.ones(self.X.shape[0],dtype=np.float32)

        if(sp.issparse(self.X)):
            # result as matrix
            self.Xsquare = self.X.multiply(self.X).sum(1)
            #result as array
            self.Xsquare = np.asarray(self.Xsquare).flatten()
        else:
            self.Xsquare =np.einsum('...i,...i',self.X,self.X)
        
        
    def clean(self,kernel_nr):
        """ clean the kernel cache """
        #self.kernel_cache.clear()

        self.bpg[kernel_nr]=0

        go = self.g_out[kernel_nr]        
        self.g_out[kernel_nr]=0
        del go
 
        gy = self.g_y[kernel_nr]
        self.g_y[kernel_nr]=0
        del gy    



    def predict_init(self, SV):
        """
        Init the classifier for prediction
        """        
        
        self.X =SV
        self.compute_diag()













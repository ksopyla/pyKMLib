# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:40:41 2014

@author: Krzysztof Sopyła
@email: krzysztofsopyla@gmail.com
@license: MIT
"""
import numpy as np
import scipy.sparse as sp
import pylru
import os
#import pycuda

import pycuda.driver as cuda
from pycuda.compiler import SourceModule        
import SparseFormats as spf

class GPURBFEll(object):
    """RBF Kernel with ellpack format"""
    
    cache_size =100
    
    Gamma=1.0
    
    #template
    func_name='rbfEllpackILPcol2multi'
    
    #template    
    module_file = os.path.dirname(__file__)+'/cu/KernelsEllpackCol2.cu'
    
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
        self.g_out=[] #gpu kernel out
        self.kernel_out=[] #cpu kernel out
        #blocks per grid for each concurrent classifier    
        self.bpg=[]
        
        #function reference
        self.func=[]
        
        #texture references for each concurrent kernel
        self.tex_ref=[]

        #main vectors 
        #gpu        
        self.g_vecI=[]
        self.g_vecJ=[]
        #cpu
        self.main_vecI=[]
        self.main_vecJ=[]    
        
        #cpu class 
        self.cls_count=[]
        self.cls=[]
        #gpu class
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
        
        #cuda initialization
        cuda.init()        
        
        self.dev = cuda.Device(0)
        self.ctx = self.dev.make_context()

        #reade cuda .cu file with module code        
        with open (self.module_file,"r") as CudaFile:
            module_code = CudaFile.read();
        
        #compile module
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
        
        
        #print 'start gpu i,j,kernel_nr ',i,j,kernel_nr
        #texReflist = list(self.tex_ref[kernel_nr])                
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

          
        
        
        


    def clean_cuda(self):
        '''
        clean all cuda resources
        '''
        
        
        for f in range(self.max_concurrent_kernels):
            
            #vecI_tex=??
            #self.g_vecI[f].free()     
            del self.g_vecI[f]

            #init texture for vector J
            #vecJ_tex=??
            #self.g_vecJ[f].free()
            del self.g_vecJ[f]
            self.g_cls_count[f].free()
            self.g_cls[f].free()
            self.g_y[f].free()
            self.g_out[f].free()

        #test it
        #del self.g_out[f] ??
        
        #copy format data structure to gpu memory
        
        self.g_val.free()
        self.g_col.free()
        self.g_len.free()
        self.g_sdot.free()
        self.g_cls_start.free()
         
        print self.ctx 
        self.ctx.pop()
        
        print self.ctx
        del self.ctx
        
        
        

    def predict_init(self, SV):
        """
        Init the classifier for prediction
        """        
        
        self.X =SV
        self.compute_diag()
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:18:51 2013

@author: ksirg
"""


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule




    

import sparse_formats as spf
import numpy as np
import scipy.sparse as sp

from sklearn import datasets
#X, Y = datasets.load_svmlight_file('Data/heart_scale')
#X, Y = datasets.load_svmlight_file('Data/toy_2d_16.train')
X, Y = datasets.load_svmlight_file('Data/w8a')

X=X.astype(np.float32)
Y=Y.astype(np.float32)

num_el,dim = X.shape
gamma = 0.5
threadsPerRow = 1
prefetch=2

from CpuSolvers import *
rbf = RBF()
rbf.gamma=gamma

rbf.init(X,Y)

i=0
j=2
vecI = X[i,:].toarray()
vecJ = X[j,:].toarray()
ki =Y[i]*Y* rbf.K_vec(vecI).flatten()
kj =Y[j]*Y*rbf.K_vec(vecJ).flatten()

kij= np.array( [ki,kj]).flatten()


v,c,r=spf.csr2ellpack(X,align=prefetch)

sd=rbf.Diag
self_dot = rbf.Xsquare
results = np.zeros(2*num_el,dtype=np.float32)

kernel_file = "KernelsEllpackCol2.cu"

with open (kernel_file,"r") as CudaFile:
    data = CudaFile.read();

#copy memory to device
g_val = cuda.to_device(v)
g_col = cuda.to_device(c)
g_r   = cuda.to_device(r)
g_self = cuda.to_device(self_dot)
g_y    = cuda.to_device(Y)
g_out = cuda.to_device(results)


#compile module
module = SourceModule(data,cache_dir='./nvcc_cache',keep=True,no_extern_c=True)

#get module function
func = module.get_function('rbfEllpackILPcol2')

#get module texture
vecI_tex=module.get_texref('VecI_TexRef')
vecJ_tex=module.get_texref('VecJ_TexRef')

#copy data to tex ref

g_vecI = cuda.to_device(vecI)
vecI_tex.set_address(g_vecI,vecI.nbytes)

g_vecJ = cuda.to_device(vecJ)
vecJ_tex.set_address(g_vecJ,vecJ.nbytes)

texList=[vecI_tex,vecJ_tex]

tpb=128#rozmiar bloku, wielokrotnosc 2

#liczba blokow 
bpg =int( np.ceil( (threadsPerRow*num_el+0.0)/tpb ))

g_num_el = np.int32(num_el)
g_i = np.int32(i)
g_j = np.int32(j)
g_gamma = np.float32(gamma)
func(g_val,g_col,g_r,g_self,g_y,g_out,g_num_el,g_i,g_j,g_gamma,block=(tpb,1,1),grid=(bpg,1),texrefs=texList)


cuda.memcpy_dtoh(results,g_out)

print "Error ",np.square(results-kij).sum()




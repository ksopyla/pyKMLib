# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:18:51 2013

@author: Krszysztof Sopyła
@email: krzysztofsopyla@gmail.com
@githubuser: ksirg
@license: MIT
"""


"""
It demostrates the usage of pycuda.

"""




import numpy as np
import scipy.sparse as sp

from sklearn import datasets
    

import sys
sys.path.append("../pyKMLib/")
import SparseFormats as spf
import Kernels as ker


#load and reorganize the dataset


#dsName = 'Data/glass.scale_binary'
dsName ='Data/w8a'
#dsName = 'Data/glass.scale.txt'
#X, Y = datasets.load_svmlight_file('Data/toy_2d_20_ones.train',dtype=np.float32)
#X, Y = datasets.load_svmlight_file('Data/toy_2d_20_order.train',dtype=np.float32)

X, Y = datasets.load_svmlight_file(dsName,dtype=np.float32)
Y=Y.astype(np.float32)

#reorder the dataset and compute class statistics
cls, idx_cls = np.unique(Y, return_inverse=True)
#contains mapped class [0,nr_cls-1]
nr_cls = cls.shape[0] 
new_classes = np.arange(0,nr_cls,dtype=np.int32)
y_map = new_classes[idx_cls]
#reorder the dataset, group class together
order =np.argsort(a=y_map,kind='mergesort')



x=X.todense()
x=x[order,:]
X = sp.csr_matrix(x)
Y=Y[order]
#print spx.data

count_cls=np.bincount(y_map).astype(np.int32)
start_cls = count_cls.cumsum()
start_cls=np.insert(start_cls,0,0).astype(np.int32)

#---------------------

num_el,dim = X.shape
gamma = 0.5
threadsPerRow = 1
prefetch=2

rbf = ker.RBF()
rbf.gamma=gamma

rbf.init(X,Y)

i=0
j=2
vecI = X[i,:].toarray()
vecJ = X[j,:].toarray()

import time
#t0=time.clock()
t0=time.time()

ki =Y[i]*Y* rbf.K_vec(vecI).flatten()
kj =Y[j]*Y*rbf.K_vec(vecJ).flatten()

#t1=time.clock()
t1=time.time()

print 'CPU RBF takes',t1-t0, 's'
kij= np.array( [ki,kj]).flatten()
print kij[0:1000:200]



##----------------------------------------------
# Ellpakc gpu kernel
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

v,c,r=spf.csr2ellpack(X,align=prefetch)

sd=rbf.Diag
self_dot = rbf.Xsquare
results = np.zeros(2*num_el,dtype=np.float32)

kernel_file = "ellpackKernel.cu"

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
#module = SourceModule(data,cache_dir='./nvcc_cache',keep=True,no_extern_c=True)

module = SourceModule(data,keep=True,no_extern_c=True,options=["--ptxas-options=-v"])

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

start_event = cuda.Event()
stop_event = cuda.Event()

start_event.record()
func(g_val,g_col,g_r,g_self,g_y,g_out,g_num_el,g_i,g_j,g_gamma,block=(tpb,1,1),grid=(bpg,1),texrefs=texList)
stop_event.record()

stop_event.synchronize()
cuTime=stop_event.time_since(start_event)


cuda.memcpy_dtoh(results,g_out)

resultsEll = np.copy(results)


print "\nEllpack time ",cuTime*1e-3
print "Error to CPU:",np.square(resultsEll-kij).sum()
print resultsEll[0:1000:200]
#print results

##------------------------------------------
# SERTILP gpu kernel


sliceSize=64
threadsPerRow=2
prefetch=2
minAlign=64 #8
v,c,r,ss=spf.csr2sertilp(X,
                         threadsPerRow=threadsPerRow, 
                         prefetch=prefetch, 
                         sliceSize=sliceSize,
                         minAlign=minAlign)

sd=rbf.Diag
self_dot = rbf.Xsquare
results = np.zeros(2*num_el,dtype=np.float32)

kernel_file = "sertilpMulti2Col.cu"

with open (kernel_file,"r") as CudaFile:
    data = CudaFile.read();
       
#compile module
#module = SourceModule(data,cache_dir='./nvcc_cache',keep=True,no_extern_c=True)
module = SourceModule(data,keep=True,no_extern_c=True,options=["--ptxas-options=-v"])
#get module function
func = module.get_function('rbfSERTILP2multi')


warp=sliceSize#32
cls1_n = count_cls[0]
align_cls1_n =  cls1_n+(warp-cls1_n%warp)%warp
cls2_n = count_cls[1]
align_cls2_n =  cls2_n+(warp-cls2_n%warp)%warp 

tpb=sliceSize*threadsPerRow#rozmiar bloku, wielokrotnosc 2
#liczba blokow 
bpg =np.ceil(((align_cls1_n+align_cls2_n)*threadsPerRow+0.0)/(tpb))
bpg=int(bpg)
#get module texture
vecI_tex=module.get_texref('VecI_TexRef')
vecJ_tex=module.get_texref('VecJ_TexRef')

#copy data to tex ref
g_vecI = cuda.to_device(vecI)
vecI_tex.set_address(g_vecI,vecI.nbytes)
g_vecJ = cuda.to_device(vecJ)
vecJ_tex.set_address(g_vecJ,vecJ.nbytes)

texList=[vecI_tex,vecJ_tex]


#copy memory to device
g_val = cuda.to_device(v)
g_col = cuda.to_device(c)
g_r   = cuda.to_device(r)
g_slice = cuda.to_device(ss)
g_self = cuda.to_device(self_dot)
g_y    = cuda.to_device(Y)
g_out = cuda.to_device(results)

g_num_el = np.int32(num_el)

align = np.ceil( 1.0*sliceSize*threadsPerRow/minAlign)*minAlign
g_align = np.int32(align)
g_i = np.int32(i)
g_j = np.int32(j)
g_i_ds= np.int32(i)
g_j_ds= np.int32(j)



       
g_cls1N_aligned = np.int32(align_cls1_n)

#gamma copy to constant memory
(g_gamma,gsize)=module.get_global('GAMMA')       
cuda.memcpy_htod(g_gamma, np.float32(gamma) )



g_cls_start = cuda.to_device(start_cls)
g_cls_count = cuda.to_device(count_cls)
g_cls = cuda.to_device(np.array([1,3],dtype=np.int32)  )


#start_event = cuda.Event()
#stop_event = cuda.Event()

start_event.record()

func(g_val,
     g_col,
     g_r,
     g_slice, 
     g_self,
     g_y,
     g_out,
     g_num_el,
     g_align, 
     g_i,
     g_j,
     g_i_ds,
     g_j_ds,
     g_cls1N_aligned,
     g_cls_start,
     g_cls_count,
     g_cls,
     block=(tpb,1,1),grid=(bpg,1),texrefs=texList)

stop_event.record()

stop_event.synchronize()

cuTime=stop_event.time_since(start_event)



cuda.memcpy_dtoh(results,g_out)


print "\nSERTILP time ",cuTime*1e-3
print "Error to CPU:",np.square(results-kij).sum()
print "Error to ELlpack:",np.square(results-resultsEll).sum()
print results[0:1000:200]

#err=results-resultsEll
#errIdx=np.where( np.abs(err)>0.0001)
#print errIdx[0].shape
#print errIdx
#
#print np.array([results[errIdx],resultsEll[errIdx]]).T

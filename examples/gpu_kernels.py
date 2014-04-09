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
#dsName ='Data/w8a'
dsName = 'Data/glass.scale.txt'
#X, Y = datasets.load_svmlight_file('Data/toy_2d_20_ones.train',dtype=np.float32)
#X, Y = datasets.load_svmlight_file('Data/toy_2d_20_order.train',dtype=np.float32)

print "Dataset: ",dsName



X, Y = datasets.load_svmlight_file(dsName,dtype=np.float32)
Y=Y.astype(np.float32)

#used for showing some elements in results array
skip= 30
#reorder the dataset and compute class statistics
cls, idx_cls = np.unique(Y, return_inverse=True)
#contains mapped class [0,nr_cls-1]
nr_cls = cls.shape[0] 
new_classes = np.arange(0,nr_cls,dtype=np.int32)
y_map = new_classes[idx_cls]
#reorder the dataset, group class together
order =np.argsort(a=y_map,kind='mergesort')


### y mapped to binary
#which class should be mapped
bin_cls = np.array([1,2]);

bin_map = np.zeros(new_classes.shape)
y_map_bin = np.zeros_like(y_map,dtype=np.float32)

y_map_bin[y_map==bin_cls[0]] =-1
y_map_bin[y_map==bin_cls[1]] =1
#first class is mapped to -1, second to 1
#bin_map[bin_cls]=np.array([-1,1])
#for i,val in enumerate(new_classes):
#    y_map_bin[y_map==i]=bin_map[i] 


x=X.todense()
x=x[order,:]
X = sp.csr_matrix(x)
Y=Y[order]
#print spx.data

count_cls=np.bincount(y_map).astype(np.int32)
start_cls = count_cls.cumsum()
start_cls=np.insert(start_cls,0,0).astype(np.int32)

i=start_cls[ bin_cls[0] ]+1
j=start_cls[ bin_cls[1] ]+1
print i,j
#---------------------

num_el,dim = X.shape
gamma = 0.5
threadsPerRow = 1
prefetch=2

rbf = ker.RBF()
rbf.gamma=gamma

rbf.init(X,Y)


vecI = X[i,:].toarray()
vecJ = X[j,:].toarray()

import time
#t0=time.clock()
t0=time.time()

#ki =Y[i]*Y* rbf.K_vec(vecI).flatten()
#kj =Y[j]*Y*rbf.K_vec(vecJ).flatten()

ki =y_map_bin[i]*y_map_bin* rbf.K_vec(vecI).flatten()
kj =y_map_bin[j]*y_map_bin*rbf.K_vec(vecJ).flatten()

#t1=time.clock()
t1=time.time()

print 'CPU RBF takes',t1-t0, 's'
kij= np.array( [ki,kj]).flatten()
print kij[0:1000:skip]



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
g_y    = cuda.to_device(y_map_bin)
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

#texture list, necessary for pycuda launch function
texList=[vecI_tex,vecJ_tex]

tpb=128#block size, power of 2

#grid size, number of blocks
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
print resultsEll[0:1000:skip]
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

#class align to sliceSize
cls_align=sliceSize
cls1_n = count_cls[0]
align_cls1_n =  cls1_n+(cls_align-cls1_n%cls_align)%cls_align
cls2_n = count_cls[1]
align_cls2_n =  cls2_n+(cls_align-cls2_n%cls_align)%cls_align 

#block size, power of 2
tpb=sliceSize*threadsPerRow
#grid size, number of blocks
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
g_y    = cuda.to_device(y_map_bin)
g_out = cuda.to_device(results)

g_num_el = np.int32(num_el)

align = np.ceil( 1.0*sliceSize*threadsPerRow/minAlign)*minAlign
g_align = np.int32(align)
g_i = np.int32(i)
g_j = np.int32(j)
g_i_ds= np.int32(i)
g_j_ds= np.int32(j)

g_cls1N_aligned = np.int32(align_cls1_n)

#gamma, copy to constant memory
(g_gamma,gsize)=module.get_global('GAMMA')       
cuda.memcpy_htod(g_gamma, np.float32(gamma) )

g_cls_start = cuda.to_device(start_cls)
g_cls_count = cuda.to_device(count_cls)


g_cls = cuda.to_device(bin_cls  )

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
print results[0:1000:skip]

#err=results-resultsEll
#errIdx=np.where( np.abs(err)>0.0001)
#print errIdx[0].shape
#print errIdx
#
#print np.array([results[errIdx],resultsEll[errIdx]]).T

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 23:07:01 2013

@author: Krszysztof Sopyła
@email: krzysztofsopyla@gmail.com

Module contains helper functios for conversion from dense and sparse format
to sparse formats utilized in pyKMLib

"""

import numpy as np
import scipy.sparse as sp
import math


def sparse_max_row(csr_mat):
    ret = np.maximum.reduceat(csr_mat.data, csr_mat.indptr[:-1])
    ret[np.diff(csr_mat.indptr) == 0] = 0
    return ret



def csr2ertilp(spmat, threadsPerRow=2, prefetch=2):
    """
    Convert scipy sparse csr matrix to ertilp format

    Parameters
    ---------------
    spmat : scipy csr sparse matrix 
        Contains data set objects, row ordered
    
    threadsPerRow : int
        How many cuda threads will be assign to one matrix row

    prefetch : int
        How many non-zero elements will be prefetch from global gpu memory

    Return
    -------------
    values : arraly_like
        numpy float32 array with nonzero values
    
    colIdx : array_like
        numpy int32 array with column index of nnz values
    
    vecNNZ : array_like
        numpy int32 array with number of nnz values per row divided by align=threadsPerRow*prefetch
    """

    assert sp.isspmatrix_csr(spmat)
    
    align = threadsPerRow*prefetch
        
    #compute maximum nonzero elements in row, 
    #max difference between two neighour index pointers in csr format
    maxRowNNZ = np.diff(spmat.indptr).max()
    
    #align max row
    rest = maxRowNNZ % align
    if(rest>0):
        maxRowNNZ=maxRowNNZ+align-rest
    
    rows,dim = spmat.shape
    
    values = np.zeros(rows*maxRowNNZ,dtype=np.float32)
    colIdx = np.zeros(rows*maxRowNNZ,dtype=np.int32)
    vecNNZ = np.zeros(rows,dtype=np.int32)
    
    for i in xrange(rows):
        vec= spmat[i,:]
                
        
        for j in xrange(vec.nnz):
            k=j/threadsPerRow
            t=j%threadsPerRow
            values[k*rows*threadsPerRow+i*threadsPerRow+t]= vec.data[j]
            colIdx[k*rows*threadsPerRow+i*threadsPerRow+t]= vec.indices[j]
        
        vecNNZ[i]= np.ceil( (vec.nnz+0.0)/align)
    
    return values,colIdx,vecNNZ

def csr2ellpack(spmat, align=1):
    """
    Convert scipy sparse csr matrix to ellpack-r format

    Parameters
    ---------------
    spmat : scipy csr sparse matrix 
        Contains data set objects, row ordered
    
    align : int
        Align of the array elements

    Return
    -------------
    values : arraly_like
        numpy float32 array with nonzero values
    
    colIdx : array_like
        numpy int32 array with column index of nnz values
    
    vecNNZ : array_like
        numpy int32 array with number of nnz values per row divided by align=threadsPerRow*prefetch
    """

    assert sp.isspmatrix_csr(spmat)
        
    #compute maximum nonzero elements in row, 
    #max difference between two neighour index pointers in csr format
    maxRowNNZ = np.diff(spmat.indptr).max()
    
    #align max row
    rest = maxRowNNZ % align
    if(rest>0):
        maxRowNNZ=maxRowNNZ+align-rest
    
    rows,dim = spmat.shape
    
    values = np.zeros(rows*maxRowNNZ,dtype=np.float32)
    colIdx = np.zeros(rows*maxRowNNZ,dtype=np.int32)
    vecNNZ = np.zeros(rows,dtype=np.int32)
    
    for i in xrange(rows):
        vec= spmat[i,:]
        for j in xrange(vec.nnz):
            values[j*rows+i]= vec.data[j]
            colIdx[j*rows+i]= vec.indices[j]
        
        vecNNZ[i]= np.ceil( (vec.nnz+0.0)/align)
    
    return values,colIdx,vecNNZ
    



def csr2sertilp(spmat, threadsPerRow=2, prefetch=2, sliceSize=64,minAlign=64):
    """
    Convert scipy sparse csr matrix to sertilp format

    Parameters
    ---------------
    spmat : scipy csr sparse matrix 
        Contains data set objects, row ordered, rows from particular class 
        should be grouped togather
    
    threadsPerRow : int
        How many cuda threads will be assign to one matrix row

    prefetch : int
        How many non-zero elements will be prefetch from global gpu memory
    
    sliceSize: int
        Determines the size of the slice, how many rows will be assigned to particular matrix strip
        
    minAlign: int
        Determines the minimum alignment

    Return
    -------------
    values : arraly_like
        numpy float32 array with nonzero values
    
    colIdx : array_like
        numpy int32 array with column index of nnz values
    
    vecNNZ : array_like
        numpy int32 array with number of nnz values per row divided by align=threadsPerRow*prefetch
    
    sliceStart: array_like
        numpy int32 array with slice start pointers
    """

    assert sp.isspmatrix_csr(spmat)
    
    rows,dim = spmat.shape    
    
    align = math.ceil( 1.0*sliceSize*threadsPerRow/minAlign)*minAlign
    
    numSlices = int(np.ceil(1.0*spmat.shape[0]/sliceSize))
    
    #slice_start=np.zeros(numSlices+1,dtype=np.int)
        
        
    #compute maximum nonzero elements in row, 
    #max difference between two neighbour index pointers in csr format
    rowLen = np.diff(spmat.indptr)
    #row lenghts divided by number of threads assign to each row and 
    #number of fetchs done by one thread 
    rowLen = np.ceil(1.0*rowLen/(threadsPerRow*prefetch)).astype(np.int32,copy=False)

    #compute max nnz in each slice
    rowDiff=np.diff(spmat.indptr)
    shapeSlice = (numSlices,sliceSize)
    #resize and fill with zeros if necessary
    rowDiff.resize(shapeSlice)
    #get max values
    maxInSlice = np.max(rowDiff,axis=1)
        
    maxInSlice=np.ceil(1.0*maxInSlice/(prefetch*threadsPerRow))*prefetch*align   
    slice_start=np.insert(np.cumsum(maxInSlice),0,0).astype(np.int32,copy=False)

    
    nnzEl = slice_start[numSlices]
    values = np.zeros(nnzEl,dtype=np.float32)
    colIdx = np.zeros(nnzEl,dtype=np.int32) #-1*np.ones(nnzEl,dtype=np.int32)
    
    for i in xrange(rows):
        
        sliceNr=i/sliceSize
        rowInSlice = i%sliceSize
        
        vec= spmat[i,:]
        for k in xrange(vec.nnz):
            rowSlice=k/threadsPerRow
            threadNr=k%threadsPerRow

            idx = slice_start[sliceNr]+align*rowSlice+rowInSlice*threadsPerRow+threadNr            
            
            values[idx]= vec.data[k]
            colIdx[idx]= vec.indices[k]
        
    
    return values,colIdx,rowLen,slice_start
    
    
    
def csr2sertilp_class(spmat,y,threadsPerRow=2, prefetch=2, sliceSize=64,minAlign=64):
    """
    Convert scipy sparse csr matrix to sertilp format with respect to class, 
    class goes to new slice, elements from different class doesnt share comon slice.
    Rows from particular class should be grouped togather
    This approach requries some excess.
    
    Example:
    slice size=64
    class count=[70 134 30 46 40]
    [slice for class1- 64 el.]
    [slice for class1 - 6el. + 58 zero rows ]
    [slice for class2 - 64 el]
    [slice for class2 - 64 el]
    [slice for class2 - 6 el.+ 58 zero rows]
    ...

    Parameters
    ---------------
    spmat : scipy csr sparse matrix 
        Contains data set objects, row ordered
    
    y: class elements, each value corespond to objects label, labels are grouped by class number 

    threadsPerRow : int
        How many cuda threads will be assign to one matrix row

    prefetch : int
        How many non-zero elements will be prefetch from global gpu memory
    
    sliceSize: int
        Determines the size of the slice, how many rows will be assigned to particular matrix strip
        
    minAlign: int
        Determines the minimum alignment

    Return
    -------------
    values : arraly_like
        numpy float32 array with nonzero values
    
    colIdx : array_like
        numpy int32 array with column index of nnz values
    
    vecNNZ : array_like
        numpy int32 array with number of nnz values per row divided by align=threadsPerRow*prefetch
    
    sliceStart: array_like
        numpy int32 array with slice start pointers
    """
    
    assert sp.isspmatrix_csr(spmat)
    
    cls, idx_cls = np.unique(y, return_inverse=True)
    #contains mapped class [0,nr_cls-1]
    nr_cls = cls.shape[0] 

    count_cls=np.bincount(y).astype(np.int32)
    start_cls = count_cls.cumsum()
    start_cls=np.insert(start_cls,0,0).astype(np.int32)

        
    
    rows,dim = spmat.shape    
    
    align = math.ceil( 1.0*sliceSize*threadsPerRow/minAlign)*minAlign
    
    #first we compute how many slices falls on a each class, than sum it    
    numSlices = int(np.ceil(1.0*count_cls/sliceSize).sum())
    
    #slice_start=np.zeros(numSlices+1,dtype=np.int)
        
        
    #compute maximum nonzero elements in row, 
    #max difference between two neighbour index pointers in csr format
    rowLen = np.diff(spmat.indptr)
    #row lenghts divided by number of threads assign to each row and 
    #number of fetchs done by one thread 
    rowLen = np.ceil(1.0*rowLen/(threadsPerRow*prefetch)).astype(np.int32,copy=False)

    #compute max nnz in each slice
    rowDiff=np.diff(spmat.indptr)
    shapeSlice = (numSlices,sliceSize)
    #resize and fill with zeros if necessary

    #get max nnz el in each slice not in class 
    maxInSlice=np.array([rowDiff[y==i].max() for i in cls])

    #split class boundaries     
    gs=np.split(rowDiff,start_cls[1:-1])
    #split in class into slice
    from itertools import izip_longest    
    sliceInClass=[list(izip_longest(*[iter(g)]*4, fillvalue=0))  for g in gs]
    sliceInClass = np.vstack(sliceInClass)
    
      #get max values
    maxInSlice = np.max(sliceInClass,axis=1)
    maxInSlice=np.ceil(1.0*maxInSlice/(prefetch*threadsPerRow))*prefetch*align   
    slice_start=np.insert(np.cumsum(maxInSlice),0,0).astype(np.int32,copy=False)

    
    nnzEl = slice_start[numSlices]
    values = np.zeros(nnzEl,dtype=np.float32)
    colIdx = np.zeros(nnzEl,dtype=np.int32) #-1*np.ones(nnzEl,dtype=np.int32)
    
    curr_cls = y[0]
    prev_cls = curr_cls    
    sliceNr=0
    rowInSlice = -1
    
    for i in xrange(rows):
        
        #increase the slice number if we already fill slice or
        #class label has changed            
        curr_cls = y[i]
        rowInSlice=rowInSlice+1 
        if rowInSlice>=sliceSize or curr_cls!=prev_cls:
            sliceNr=sliceNr+1
            rowInSlice=0
            
        prev_cls = curr_cls    
        
        
        #row number in particular slice            
        
        #take the i-th row
        vec= spmat[i,:]
        #compute i-th vector nnz elements position in arrays
        for k in xrange(vec.nnz):
            rowSlice=k/threadsPerRow
            threadNr=k%threadsPerRow

            idx = slice_start[sliceNr]+align*rowSlice+rowInSlice*threadsPerRow+threadNr            
            
            values[idx]= vec.data[k]
            colIdx[idx]= vec.indices[k]
        
    
    return values,colIdx,rowLen,slice_start
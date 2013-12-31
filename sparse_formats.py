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
    
    values= np.array([0])
    colIdx = np.array([0])
    vecNNZ = np.array([0])
    
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

def sp2sertilp(matrix):
    pass    
    


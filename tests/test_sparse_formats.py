# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 23:12:01 2014

@author: Krszysztof Sopyła
@email: krzysztofsopyla@gmail.com
@license: MIT
"""

import unittest

import numpy as np
import scipy.sparse as sp
from sklearn import datasets
import sys
sys.path.append("../pyKMLib/")
import SparseFormats as spf
 
class TestSparseFormats(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_csr2sertilp(self):

        mat = np.array([ [1,0,2,0,3,0], 
                         [4,0,5,0,0,0],
                         [0,0,0,6,7,0],
                         [0,0,0,0,0,8],
                         [21,0,22,0,23,0], 
                         [24,0,25,0,0,0],
                         [0,0,0,26,27,0],
                         [0,0,0,0,0,28]
                       ])
        
        sp_mat = sp.csr_matrix(mat)
        
        row_len_right = np.array([1,1,1,1,1,1,1,1])
        sl_start_right = np.array([0,16,32])
        val_right = np.array([1.0,2.0,4.0,5.0,6.0,7.0,8.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,21.0,22.0,24.0,25.0,26.0,27.0,28.0,0.0,23.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        #collumns taken directly from dataset, 
        col_vs_right = np.array([1,3,1,3,4,5,6,0,5,0,0,0,0,0,0,0,1,3,1,3,4,5,6,0,5,0,0,0,0,0,0,0])
        #but in sparse format collumns start from 0  so we have to substract 1      
        col_right = col_vs_right-1
        col_right[col_right==-1]=0
                
        val,col,row_len,sl_start=spf.csr2sertilp(sp_mat,
                                            threadsPerRow=2, 
                                            prefetch=2,
                                            sliceSize=4,
                                            minAlign=2*4)
                                                    
        self.assertTrue(np.allclose(row_len,row_len_right), 'sliced ellpack row length arrays are not equal')
        self.assertTrue(np.allclose(sl_start,sl_start_right), 'sliced ellpack slice start arrays are not equal')       
        self.assertTrue(np.allclose(val,val_right), 'sliced ellpack values arrays are not equal')
        self.assertTrue(np.allclose(col,col_right), 'sliced ellpack collumns arrays are not equal')
       
    def test_csr2sertilp_class_smaller_than_slice_size(self):
        
        threadsPerRow=2
        prefetch=2
        sliceSize=4
        minAlign=2*4

        mat = np.array([ [1,0,2,0,3,0], 
                         [4,0,5,0,0,0],
                         [0,0,0,6,7,0],
                         [0,0,0,0,0,8],
                         [9,0,10,0,11,0], 
                         [12,0,13,0,0,0],
                         [0,0,0,14,15,0],
                         [0,0,0,0,0,16]
                       ])
        y = np.array([0,0,0,1,1,2,2,2])
        
        sp_mat = sp.csr_matrix(mat)
        row_len_right = np.array([1,1,1,1,1,1,1,1])        
        sl_start_right = np.array([0,16,32, 48])
        cls_slice_right = np.array([0,1,2,3])
        
        val_right = np.array([1.0,2.0, 4.0,5.0, 6.0,7.0, 0.0,0.0,
                              3.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
                              8.0,0.0, 9.0,10.0, 0.0,0.0, 0.0,0.0,
                              0.0,0.0, 11.0,0.0, 0.0,0.0, 0.0,0.0,
                              12.0,13.0, 14.0,15.0, 16.0,0.0, 0.0,0.0,
                              0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,                              
                              ])
        col_right = np.array([0,2,  0,2,  3,4,  0,0,
                                 4,0,  0,0,  0,0,  0,0,
                                 5,0,  0,2,  0,0,  0,0,
                                 0,0,  4,0,  0,0,  0,0,
                                 0,2,  3,4,  5,0,  0,0, 
                                 0,0,  0,0,  0,0,  0,0])                              
        
        val,col,row_len,sl_start,cls_slice=spf.csr2sertilp_class(sp_mat,y,
                                            threadsPerRow=threadsPerRow, 
                                            prefetch=prefetch,
                                            sliceSize=sliceSize,
                                            minAlign=minAlign)
                                                    
        self.assertTrue(np.allclose(row_len,row_len_right), 'sliced ellpack row length arrays are not equal')
        self.assertTrue(np.allclose(sl_start,sl_start_right), 'sliced ellpack slice start arrays are not equal')       
        self.assertTrue(np.allclose(cls_slice,cls_slice_right), 'sliced ellpack class slice start arrays are not equal')       
        self.assertTrue(np.allclose(val,val_right), 'sliced ellpack values arrays are not equal')
        self.assertTrue(np.allclose(col,col_right), 'sliced ellpack collumns arrays are not equal')
       
    def test_csr2sertilp_class_grather_than_slice_size(self):
        
        threadsPerRow=2
        prefetch=2
        sliceSize=4
        minAlign=2*4

        mat = np.array([ [1,0,2,0,3,0], 
                         [1,2,0,0,0,0],
                         [1,2,3,4,0,0],  
                         [4,0,5,0,0,0],
                         [0,0,0,6,7,0],
                         [0,0,0,0,0,8],
                         [9,0,10,0,11,0], 
                         [12,0,13,0,0,0],
                         [0,0,0,14,15,0],
                         [0,0,0,0,0,16]
                       ])
        y = np.array([0,0,0,0,0,1,1,2,2,2])
        
        sp_mat = sp.csr_matrix(mat)
        row_len_right = np.array([1,1,1,1,1,1,1,1,1,1])        
        sl_start_right = np.array([0,16,32,48,64])
        cls_slice_right = np.array([0,2,3,4])
        
        val_right = np.array([1.0,2.0, 1.0,2.0, 1.0,2.0, 4.0,5.0, 
                              3.0,0.0, 0.0,0.0, 3.0,4.0, 0.0,0.0,
                              6.0,7.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
                              0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,                              
                              8.0,0.0, 9.0,10.0, 0.0,0.0, 0.0,0.0,
                              0.0,0.0, 11.0,0.0, 0.0,0.0, 0.0,0.0,
                              12.0,13.0, 14.0,15.0, 16.0,0.0, 0.0,0.0,
                              0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,                              
                              ])
        col_right = np.array([0,2,  0,1,  0,1,  0,2,
                              4,0,  0,0,  2,3,  0,0,
                              3,4,  0,0,  0,0,  0,0,
                              0,0,  0,0,  0,0,  0,0,        
                              5,0,  0,2,  0,0,  0,0,
                              0,0,  4,0,  0,0,  0,0,
                              0,2,  3,4,  5,0,  0,0, 
                              0,0,  0,0,  0,0,  0,0])                              
        
        val,col,row_len,sl_start, cls_slice=spf.csr2sertilp_class(sp_mat,y,
                                            threadsPerRow=threadsPerRow, 
                                            prefetch=prefetch,
                                            sliceSize=sliceSize,
                                            minAlign=minAlign)
                                                    
        self.assertTrue(np.allclose(row_len,row_len_right), 'sliced ellpack row length arrays are not equal')
        self.assertTrue(np.allclose(sl_start,sl_start_right), 'sliced ellpack slice start arrays are not equal')       
        self.assertTrue(np.allclose(cls_slice,cls_slice_right), 'sliced ellpack class slice start arrays are not equal')       
        self.assertTrue(np.allclose(val,val_right), 'sliced ellpack values arrays are not equal')
        self.assertTrue(np.allclose(col,col_right), 'sliced ellpack collumns arrays are not equal')
       
                
 
 
if __name__ == '__main__':
    unittest.main(exit=False)
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 06 16:01:51 2013

@author: ksirg
"""



from GPUSolvers import *

import numpy as np
import scipy.sparse as sp
import time

import pylab as pl

from sklearn import datasets
# import some data to play with
iris = datasets.load_iris()

#X = iris.data
#Y = iris.target
#

# multiclass 
#X, Y = datasets.load_svmlight_file('Data/glass.scale.txt')
#X, Y = datasets.load_svmlight_file('glass.scale_3cls.txt')

#binary
#X, Y = datasets.load_svmlight_file('glass.scale_binary')
X, Y = datasets.load_svmlight_file('Data/heart_scale')
#X, Y = datasets.load_svmlight_file('Data/w8a')

#X, Y = datasets.load_svmlight_file('toy_2d_16.train')

C=1

from sklearn import svm

#clf = svm.SVC(C=C,kernel='linear',verbose=True)
clf = svm.SVC(C=C,kernel='rbf',gamma=1.0,verbose=True)
t0=time.clock()
svm_m= clf.fit(X,Y)
t1=time.clock()

print '\nTrains Takes: ', t1-t0
#print 'alpha\n',clf.dual_coef_.toarray()

#print 'nSV=',clf.n_support_
#print 'sv \n',clf.support_vectors_.toarray()
#print 'sv idx=',clf.support_


t0=time.clock()
pred1 = clf.predict(X)
t1=time.clock()
print '\nPredict Takes: ', t1-t0
#print pred1
acc = (0.0+sum(Y==pred1))/len(Y)

print 'acc=',acc

print '--------------\n'


#np.random.seed(0)
#n=6
#X = np.random.randn(n, 2)
#Y = np.random.randint(1,4,n)
#X = np.array([ (1,2), (3,4), (5,6), (7,8), (9,0)])
#Y = np.array([4,1,2,1,4])

svm_solver =  GPUSVM2Col(X,Y,C)
#kernel = Linear()
kernel = GPURBF()


t0=time.clock()
svm_solver.init(kernel)
t1=time.clock()
print '\nInit takes',t1-t0

t0=time.clock()

svm_solver.train()

t1=time.clock()

print '\nTakes: ', t1-t0

for k in xrange(len(svm_solver.models)):
    m=svm_solver.models[k]
    print 'Iter=',m.Iter
    print 'Obj={} Rho={}'.format(m.Obj,m.Rho)

    print 'nSV=',m.NSV
    #print m.Alpha


t0=time.clock()
pred2,dec_vals=svm_solver.predict(X)
t1=time.clock()
print '\nPredict Takes: ', t1-t0
#print pred2
acc = (0.0+sum(Y==pred2))/len(Y)

print 'acc=',acc

#x1=X[:,0].toarray()
#x2=X[:,1].toarray()
#pl.scatter(x1,x2,c=Y,cmap=pl.cm.Paired, s=80)
#pl.show()



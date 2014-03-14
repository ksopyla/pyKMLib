pyKMLib
=======

KMLib in python. Kernel SVM method accelerated with CUDA.
Library allows for classification sparse dataset with use of different sprase storage (matrix) format.
CUDA SVM in python.

It is a partial python port of .net KMLib project https://github.com/ksirg/KMLib 



Prerequisits
-------------
* Python 2.7
* Numpy 1.7 MKL
* Scipy
* Numba
* pycuda 2013.1.1


Ubuntu 13.10 prerequisits installation
-----------

**llvm** - sudo apt-get install llvm

**llvmpy** - 

wget https://github.com/llvmpy/llvmpy/releases/tag/0.12.3

tar zxvf 0.12.3.tar.gz

cd 0.12.3

sudo LLVM_CONFIG_PATH=/usr/bin/llvm-config python setup.py install

**numba** - sudo pip install numba

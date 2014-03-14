pyKMLib
=======

KMLib in python. Kernel SVM method accelerated with CUDA.
Library allows for classification sparse dataset with use of different sprase storage (matrix) format.
CUDA SVM in python.

It is a partial python port of .net KMLib project https://github.com/ksirg/KMLib 



Prerequisits
-------------
* Python 2.7
* pycuda 2013.1.1
* Numpy 1.7 MKL
* Scipy
* Numba


Ubuntu 13.10 prerequisits installation
-----------

##numba installation

* *llvm* - This install llvm 3.4
```sh
 sudo apt-get install llvm
```

* *llvmpy* - python llvm wrapper

```sh
wget https://github.com/llvmpy/llvmpy/releases/tag/0.12.3
tar zxvf 0.12.3.tar.gz
cd 0.12.3
sudo LLVM_CONFIG_PATH=/usr/bin/llvm-config python setup.py install
```

* *numba* - 
```sh
sudo pip install numba
```

**pycuda installation**

*Warning!*

*sudo apt-get install pycuda* - probably override your nvidia driver installation, so If you previously install nvidia driver and cuda toolkit previously than it is not recomended. (I have install cuda toolkit and driver with help http://askubuntu.com/questions/380609/anyone-has-successfully-installed-cuda-5-5-on-ubuntu-13-10-64-bit )

```sh
vim ~/.bashrc 
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=${CUDA_HOME}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

sudo PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH pip install pycuda
```

# Julia set generator

This is an implementation of the interactive Julia set generator written in Python. The implementation supports changes of the size of the complex plane that is being observed. The main part of the computations is done on the GPU, using CUDA 9.0. 

Software requirements to run the code:
1. Python 3.6.1
2. CUDA 9.0
3. Python modules
	* PyCuda
	* Rkinter
	* PIL
	* NumPy
	* CMath

The examples of the generated sets are:

![julia_1](https://github.com/superkirill/julia_set/blob/master/Julia_1.jpg?raw=true)
![julia_2](https://github.com/superkirill/julia_set/blob/master/Julia_2.jpg?raw=true)

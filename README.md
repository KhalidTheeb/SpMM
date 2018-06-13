## SpMM
Sparse matrix multi vector multiplication

This project was based on CUSP, some code is reused from the ell-spmv available in CUSP.

Related publication available here: 
http://ieeexplore.ieee.org/abstract/document/7056883/?reload=true


## Compilation:
```
nvcc ELL.cu mmio.c -o ell_SpMM
```

## Sample Output:
```
Using 64-bit floating point precision

Reading sparse matrix from file (/scratch/cant.mtx): done
Using 62451-by-62451 matrix with 4007383 nonzero values
###   Testing the performance of SpMM using ELL   ###
Number of vectors 2    2
	benchmarking ell                  [gpu]:   0.0791 ms ( 202.63 GFLOP/s)
	benchmarking ell                  [gpu]: ( 1113.24 Gbytes/s)
###   Testing the performance of SpMM using ELL   ###
Number of vectors 4    4
	benchmarking ell                  [gpu]:   0.1056 ms ( 303.60 GFLOP/s)
	benchmarking ell                  [gpu]: ( 833.99 Gbytes/s)
###   Testing the performance of SpMM using ELL   ###
Number of vectors 8    8
	benchmarking ell                  [gpu]:   0.1498 ms ( 427.89 GFLOP/s)
	benchmarking ell                  [gpu]: ( 587.70 Gbytes/s)
###   Testing the performance of SpMM using ELL   ###
Number of vectors 16    16
	benchmarking ell                  [gpu]:   0.2959 ms ( 433.43 GFLOP/s)
	benchmarking ell                  [gpu]: ( 297.66 Gbytes/s)
###   Testing the performance of SpMM using ELL   ###
Number of vectors 32    32
	benchmarking ell                  [gpu]:   0.6233 ms ( 411.45 GFLOP/s)
	benchmarking ell                  [gpu]: ( 141.28 Gbytes/s)
```

/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */



#pragma once
#include <stdio.h>

#include "sparse_formats.h"
#include "timer.h"
 
template <typename IndexType, typename ValueType>
size_t bytes_per_spmv(const ell_matrix<IndexType,ValueType>& mtx)
{
	    
    size_t bytes = 0;
    bytes += 1*sizeof(IndexType) * mtx.num_nonzeros; // column index
    bytes += 1*sizeof(ValueType) * mtx.stride * mtx.num_cols_per_row; // A[i,j] and padding
    bytes += 1*sizeof(ValueType) * mtx.num_nonzeros; // x[j]
    bytes += 2*sizeof(ValueType) * mtx.num_rows;     // y[i] = y[i] + ...
    return bytes;
}
  


template <typename IndexType, typename ValueType, typename SpMM>
void benchmark_ell(const csr_matrix<IndexType,ValueType>& csr, SpMM spmm, const memory_location loc, const char * method_name, const size_t min_iterations = 1, const size_t max_iterations = 1000, const double seconds = 3.0)
{


    for (int NUMVECTORS=2; NUMVECTORS<=32; NUMVECTORS*=2){
	//for (int VECBLOCK=8; VECBLOCK<=NUMVECTORS; VECBLOCK*=2){


    // initialize host vectors
    ValueType * x_host = new_host_array<ValueType>(csr.num_cols* NUMVECTORS);

     for(IndexType j = 0; j < NUMVECTORS ; j++)
	    for(IndexType i = 0; i < csr.num_cols; i++)
       	 x_host[j*csr.num_cols+i] = rand() / (RAND_MAX + 1.0);

    ValueType * y_host = new_host_array<ValueType>(csr.num_rows*NUMVECTORS);
    std::fill(y_host, y_host + csr.num_rows*NUMVECTORS, 0);
    ell_matrix<IndexType,ValueType> ell_device ;
    ell_matrix<IndexType,ValueType> ell;
    //initialize device arrays
    
    ValueType * y_loc = copy_array(y_host, csr.num_rows*NUMVECTORS, HOST_MEMORY, loc);
    ValueType * x_loc = copy_array(x_host, csr.num_cols*NUMVECTORS , HOST_MEMORY, loc);

    printf("###   Testing the performance of SpMM using ELL   ###\n");
    printf("Number of vectors %d    %d\n", NUMVECTORS, NUMVECTORS);//VECBLOCK);
    size_t num_iterations = max_iterations;

	
    IndexType max_cols_per_row = static_cast<IndexType>( (3 * csr.num_nonzeros) / csr.num_rows + 1 );//khalid: equation is for dia
    ell = csr_to_ell<IndexType,ValueType>(csr, max_cols_per_row);
    if (ell.num_nonzeros == 0 && csr.num_nonzeros != 0){
       return;
    }

    ell_device= copy_matrix_to_device(ell);

    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        spmm(ell_device, x_loc, y_loc, NUMVECTORS, NUMVECTORS);//VECBLOCK);
    cudaThreadSynchronize();
    double msec_per_iteration = t.milliseconds_elapsed() / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (NUMVECTORS *2.0 * (double) ell.num_nonzeros / sec_per_iteration) / 1e9;
	double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(ell) / sec_per_iteration) / 1e9;
	
	
    const char * location = (loc == HOST_MEMORY) ? "cpu" : "gpu";
	printf("\tbenchmarking %-20s [%s]: %8.4f ms ( %5.2f GFLOP/s)\n", \
            method_name, location, msec_per_iteration, GFLOPs);    //deallocate buffers
	printf("\tbenchmarking %-20s [%s]: ( %5.2f Gbytes/s)\n", \
			method_name, location, GBYTEs);    //deallocate buffers		
			
			
			
			
    delete_device_matrix(ell_device);
    delete_host_matrix(ell);
    delete_host_array(y_host);
    delete_host_array(x_host);
    delete_array(y_loc, loc);
    delete_array(x_loc, loc);

//}
}

}


template <typename IndexType, typename ValueType, typename SpMM>
void benchmark_ell_on_device(const csr_matrix<IndexType,ValueType>& csr, SpMM spmm, const char * method_name = NULL)
{
    benchmark_ell<IndexType,ValueType,SpMM>(csr, spmm, DEVICE_MEMORY, method_name);
}


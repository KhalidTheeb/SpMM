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

// Functions to test spmm kernels

#include <algorithm>
#include <limits>
#include <cmath>
#include "mem.h"
#include "spmm_host.h"
#include "spmm_ell_device.cu.h"

template <typename T>
T maximum_relative_error(const T * A, const T * B, const size_t N, size_t NUMVECTORS)
{
    T max_error = 0;
    float max_absolute_error;
    int number_of_errors=0;
    int vector=0;
    int location=0;
    T eps = std::sqrt( std::numeric_limits<T>::epsilon() );
    
    for (size_t j=0; j<NUMVECTORS ; j++){
    	for(size_t i = 0; i < N; i++)
    	{
       	const T a = A[j*N+i];
        	const T b = B[j*N+i];
        	const T error = std::abs(a - b);
        	if (error != 0){
                     if ( error/(std::abs(a) + std::abs(b) + eps) > 5 * std::sqrt( std::numeric_limits<T>::epsilon() ) )number_of_errors++;
            		max_error = std::max(max_error, error/(std::abs(a) + std::abs(b) + eps) );
			if (error/(std::abs(a) + std::abs(b) + eps) == max_error) {max_absolute_error=std::abs(a - b) ; vector=j; location=i;}
        	}
    	}
    }
    printf("number of errors = %d\n", number_of_errors);
    printf("location of maximum error= %d in vector %d error %f\n", location, vector, max_absolute_error);
    return max_error;
}


template <typename IndexType, typename ValueType>
void test_spmm_ell_kernel(const csr_matrix<IndexType,ValueType>& csr)
{

    printf("\n####  Testing ELL SpMM Kernel ####\n");
   
    for (int NUMVECTORS=2; NUMVECTORS<=32; NUMVECTORS*=2){
	for (int VECBLOCK=2; VECBLOCK<=NUMVECTORS; VECBLOCK*=2){



    const IndexType num_rows = csr.num_rows;
    const IndexType num_cols = csr.num_cols;

    // Initialize host vectors
    ValueType * x_host = new_host_array<ValueType>(num_cols* NUMVECTORS );
    ValueType * y_host = new_host_array<ValueType>(num_rows* NUMVECTORS );

    
    for(IndexType j = 0; j < NUMVECTORS ; j++)
	    for(IndexType i = 0; i < num_cols; i++)
       	 x_host[j*num_cols+i] = rand() / (RAND_MAX + 1.0); 
    for(IndexType j = 0; j < NUMVECTORS ; j++)
	    for(IndexType i = 0; i < num_rows; i++)
       	 y_host[j*num_rows+i] = rand() / (RAND_MAX + 1.0);
 
   
    printf("Creating ELL_matrix.....\n");
    IndexType max_cols_per_row = static_cast<IndexType>( (3 * csr.num_nonzeros) / csr.num_rows + 1 );//khalid: equation is for dia
    ell_matrix<IndexType,ValueType> ell = csr_to_ell<IndexType,ValueType>(csr, max_cols_per_row);
    if (ell.num_nonzeros == 0 && csr.num_nonzeros != 0){
       printf(" num_cols_per_row (%d) excedes limit (%d)\n", ell.num_cols_per_row, max_cols_per_row);
       return;
    }
    printf(" found %d num_cols_per_row\n", ell.num_cols_per_row);
    printf("done\n");
 

   printf("###   Checking the correctness of ELL kernel   ###\n");    
   // transfer matrices from host to destination location
   ell_matrix<IndexType,ValueType> sm2_loc2 = copy_matrix_to_device(ell);
   printf("Finished copying the matrix to device memory...\n");
    
   // create vectors in appropriate locations
   ValueType * x_loc1 = copy_array(x_host, num_cols*NUMVECTORS , HOST_MEMORY, HOST_MEMORY);
   ValueType * x_loc2 = copy_array(x_host, num_cols*NUMVECTORS , HOST_MEMORY, DEVICE_MEMORY);
   ValueType * y_loc1 = copy_array(y_host, num_rows*NUMVECTORS , HOST_MEMORY, HOST_MEMORY);
   ValueType * y_loc2 = copy_array(y_host, num_rows*NUMVECTORS , HOST_MEMORY, DEVICE_MEMORY); 
   // compute y = A*x
   
   

   printf("Calling CSR kernel on host....\n");
   for(IndexType j = 0; j < NUMVECTORS ; j++)
   	spmv_csr_serial_host<IndexType,ValueType>(csr, x_loc1 + j*num_cols, y_loc1 + j*num_rows);
   printf("done...\n");
   printf("Calling ELL kernel on device.....\n");
   //spmm_ell_device(sm2_loc2,  x_loc2, y_loc2);
   //spmm_ell_tex_device(sm2_loc2,  x_loc2, y_loc2);
   printf("done...\n ");

   
   // transfer results to host
   ValueType * y_sm1_result = copy_array(y_loc1, num_rows*NUMVECTORS , HOST_MEMORY, HOST_MEMORY);
   ValueType * y_sm2_result = copy_array(y_loc2, num_rows*NUMVECTORS , DEVICE_MEMORY, HOST_MEMORY);

   ValueType max_error = maximum_relative_error(y_sm1_result, y_sm2_result, num_rows, NUMVECTORS );
   printf("[max error %9f]", max_error);
    
    if ( max_error > 5 * std::sqrt( std::numeric_limits<ValueType>::epsilon() ) )
       printf(" POSSIBLE FAILURE");
    printf("\n");    
    // cleanup
    delete_device_matrix(sm2_loc2);
    delete_host_matrix(ell);
    delete_host_array(x_host);
    delete_host_array(y_host);
    delete_array(x_loc1, HOST_MEMORY);
    delete_array(x_loc2, DEVICE_MEMORY);
    delete_array(y_loc1, HOST_MEMORY);
    delete_array(y_loc2, DEVICE_MEMORY);
    delete_host_array(y_sm1_result);
    delete_host_array(y_sm2_result);



}}

}


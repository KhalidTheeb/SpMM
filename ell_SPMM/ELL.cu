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


#include <iostream>
#include <stdio.h>
#include "cmdline.h"
#include "sparse_io.h"
#include "sparse_formats.h"
#include "test_spmm.h"
#include "benchmark_ell.h"
#include "spmm_ell_device.cu.h"

template <typename IndexType, typename ValueType>
void test_ell_matrix_kernel(const csr_matrix<IndexType,ValueType>& csr)
{
 
   //Test the performance of ell kernel
   benchmark_ell_on_device(csr, spmm_ell_device<IndexType, ValueType>,"ell");

}

template <typename IndexType, typename ValueType>
void run_ell(int argc, char **argv)
{
    char * mm_filename = NULL;
    for(int i = 1; i < argc; i++){
        if(argv[i][0] != '-'){
            mm_filename = argv[i];
            break;
        }
    }
    

    csr_matrix<IndexType,ValueType> csr;

    csr= read_csr_matrix<IndexType,ValueType>(mm_filename);
            

    printf("Using %d-by-%d matrix with %d nonzero values\n", csr.num_rows, csr.num_cols, csr.num_nonzeros); 

    // fill matrix with random values: some matrices have extreme values, 
    // which makes correctness testing difficult, especially in single precision
    srand(13);
    for(IndexType i = 0; i < csr.num_nonzeros; i++){
      csr.Ax[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
    }
    
    // Call the function that tests the correctness and performance of ell kernel
    test_ell_matrix_kernel(csr);
    
    delete_host_matrix(csr);
}

int main(int argc, char** argv)
{
    int precision = 64;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);
    printf("\nUsing %d-bit floating point precision\n\n", precision);

    if(precision ==  32){
        run_ell<unsigned int, float>(argc,argv);
    }
    else if(precision == 64)
    {
        int current_device = -1;
        cudaDeviceProp properties;
        cudaGetDevice(&current_device);
        cudaGetDeviceProperties(&properties, current_device);
        if (properties.major == 1 && properties.minor < 3)
            std::cerr << "ERROR: Support for \'double\' requires Compute Capability 1.3 or greater\n\n";
        else
        run_ell<unsigned int, double>(argc,argv);
    }
   
    return EXIT_SUCCESS;
}


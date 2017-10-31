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

#include "sparse_formats.h"
#include "utils.h"
#include "texture.h"



template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel2(const IndexType num_rows, 
                 const IndexType num_cols, 
                 const IndexType num_cols_per_row,
                 const IndexType stride,
                 const IndexType * Aj,
                 const ValueType * Ax, 
                 const ValueType * x, 
                       ValueType * y)
{
   const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

    ValueType sum1 = y[row];
	ValueType sum2 = y[row+num_rows];
	
    Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
            sum1 += A_ij * fetch_x<UseCache>(col, x);
			sum2 += A_ij * fetch_x<UseCache>(col+num_cols, x);
        }

        Aj += stride;
        Ax += stride;
    }

    y[row] = sum1;
	y[row+num_rows] = sum2;
}

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel3(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax, 
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

    ValueType sum1 = y[row];
	ValueType sum2 = y[row+num_rows];
	ValueType sum3 = y[row+num_rows+num_rows];
	 
    Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
            sum1 += A_ij * fetch_x<UseCache>(col, x);
			sum2 += A_ij * fetch_x<UseCache>(col+num_cols, x);
			sum3 += A_ij * fetch_x<UseCache>(col+num_cols+num_cols, x);
		}

        Aj += stride;
        Ax += stride;
    }

    y[row] = sum1;
	y[row+num_rows] = sum2;
	y[row+num_rows+num_rows] = sum3;
}

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel4(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax, 
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

	int temp=row;
    ValueType sum1 = y[row];
    temp += num_rows;
    ValueType sum2 = y[temp];
    temp += num_rows;
    ValueType sum3 = y[temp];
    temp += num_rows;
    ValueType sum4 = y[temp];
	
	Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
           	temp=col;
            sum1 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum2 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum3 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum4 += A_ij * fetch_x<UseCache>(temp, x);
		}

        Aj += stride;
        Ax += stride;
    }

    temp=row;
    y[row] = sum1;
    temp += num_rows;
    y[temp] = sum2;
    temp += num_rows;
    y[temp] = sum3;
    temp += num_rows;
    y[temp] =sum4;
}

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel5(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

	int temp=row;
    ValueType sum1 = y[row];
    temp += num_rows;
    ValueType sum2 = y[temp];
    temp += num_rows;
    ValueType sum3 = y[temp];
    temp += num_rows;
    ValueType sum4 = y[temp];
	temp += num_rows;
    ValueType sum5 = y[temp];
	
	Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
           	temp=col;
            sum1 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum2 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum3 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum4 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum5 += A_ij * fetch_x<UseCache>(temp, x);
		}

        Aj += stride;
        Ax += stride;
    }

    temp=row;
    y[row] = sum1;
    temp += num_rows;
    y[temp] = sum2;
    temp += num_rows;
    y[temp] = sum3;
    temp += num_rows;
    y[temp] =sum4;
	temp += num_rows;
    y[temp] =sum5;
}

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel6(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

	int temp=row;
    ValueType sum1 = y[row];
    temp += num_rows;
    ValueType sum2 = y[temp];
    temp += num_rows;
    ValueType sum3 = y[temp];
    temp += num_rows;
    ValueType sum4 = y[temp];
	temp += num_rows;
    ValueType sum5 = y[temp];
	temp += num_rows;
    ValueType sum6 = y[temp];
	
	Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
           	temp=col;
            sum1 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum2 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum3 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum4 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum5 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum6 += A_ij * fetch_x<UseCache>(temp, x);
		}

        Aj += stride;
        Ax += stride;
    }

    temp=row;
    y[row] = sum1;
    temp += num_rows;
    y[temp] = sum2;
    temp += num_rows;
    y[temp] = sum3;
    temp += num_rows;
    y[temp] =sum4;
	temp += num_rows;
    y[temp] =sum5;
	temp += num_rows;
    y[temp] =sum6;
}

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel8(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x, 
                      ValueType * y)
{
     const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

	int temp=row;
    ValueType sum1 = y[row];
    temp += num_rows;
    ValueType sum2 = y[temp];
    temp += num_rows;
    ValueType sum3 = y[temp];
    temp += num_rows;
    ValueType sum4 = y[temp];
	temp += num_rows;
    ValueType sum5 = y[temp];
	temp += num_rows;
    ValueType sum6 = y[temp];
	temp += num_rows;
	ValueType sum7 = y[temp];
    temp += num_rows;
    ValueType sum8 = y[temp];
	
	Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
           	temp=col;
            sum1 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum2 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum3 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum4 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum5 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum6 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum7 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum8 += A_ij * fetch_x<UseCache>(temp, x);
		}

        Aj += stride;
        Ax += stride;
    }

    temp=row;
    y[row] = sum1;
    temp += num_rows;
    y[temp] = sum2;
    temp += num_rows;
    y[temp] = sum3;
    temp += num_rows;
    y[temp] =sum4;
	temp += num_rows;
    y[temp] =sum5;
	temp += num_rows;
    y[temp] =sum6;
	temp += num_rows;
    y[temp] = sum7;
    temp += num_rows;
    y[temp] =sum8;
}

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel10(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

	int temp=row;
    ValueType sum1 = y[row];
    temp += num_rows;
    ValueType sum2 = y[temp];
    temp += num_rows;
    ValueType sum3 = y[temp];
    temp += num_rows;
    ValueType sum4 = y[temp];
	temp += num_rows;
    ValueType sum5 = y[temp];
	temp += num_rows;
    ValueType sum6 = y[temp];
	temp += num_rows;
	ValueType sum7 = y[temp];
    temp += num_rows;
    ValueType sum8 = y[temp];
	temp += num_rows;
    ValueType sum9 = y[temp];
    temp += num_rows;
    ValueType sum10 = y[temp];
	
	Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
           	temp=col;
            sum1 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum2 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum3 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum4 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum5 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum6 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum7 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum8 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum9 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum10 += A_ij * fetch_x<UseCache>(temp, x);
		}

        Aj += stride;
        Ax += stride;
    }

    temp=row;
    y[row] = sum1;
    temp += num_rows;
    y[temp] = sum2;
    temp += num_rows;
    y[temp] = sum3;
    temp += num_rows;
    y[temp] =sum4;
	temp += num_rows;
    y[temp] =sum5;
	temp += num_rows;
    y[temp] =sum6;
	temp += num_rows;
    y[temp] = sum7;
    temp += num_rows;
    y[temp] =sum8;
	temp += num_rows;
    y[temp] = sum9;
    temp += num_rows;
    y[temp] =sum10;

}
template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel16(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType stride,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType row = large_grid_thread_id();

    if(row >= num_rows){ return; }

	int temp=row;
    ValueType sum1 = y[row];
    temp += num_rows;
    ValueType sum2 = y[temp];
    temp += num_rows;
    ValueType sum3 = y[temp];
    temp += num_rows;
    ValueType sum4 = y[temp];
	temp += num_rows;
    ValueType sum5 = y[temp];
	temp += num_rows;
    ValueType sum6 = y[temp];
	temp += num_rows;
	ValueType sum7 = y[temp];
    temp += num_rows;
    ValueType sum8 = y[temp];
	temp += num_rows;
    ValueType sum9 = y[temp];
    temp += num_rows;
    ValueType sum10 = y[temp];
	temp += num_rows;
	ValueType sum11 = y[temp];
    temp += num_rows;
    ValueType sum12 = y[temp];
    temp += num_rows;
    ValueType sum13 = y[temp];
    temp += num_rows;
    ValueType sum14 = y[temp];
	temp += num_rows;
    ValueType sum15 = y[temp];
	temp += num_rows;
    ValueType sum16 = y[temp];
	
	
	Aj += row;
    Ax += row;

    for(IndexType n = 0; n < num_cols_per_row; n++){
        const ValueType A_ij = *Ax;

        if (A_ij != 0){
            const IndexType col = *Aj;
           	temp=col;
            sum1 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum2 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum3 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum4 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum5 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum6 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum7 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum8 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum9 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum10 += A_ij * fetch_x<UseCache>(temp, x);	
			temp += num_cols;
			sum11 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum12 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum13 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
			sum14 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum15 += A_ij * fetch_x<UseCache>(temp, x);
			temp += num_cols;
			sum16 += A_ij * fetch_x<UseCache>(temp, x);
		}

        Aj += stride;
        Ax += stride;
    }

    temp=row;
    y[row] = sum1;
    temp += num_rows;
    y[temp] = sum2;
    temp += num_rows;
    y[temp] = sum3;
    temp += num_rows;
    y[temp] =sum4;
	temp += num_rows;
    y[temp] =sum5;
	temp += num_rows;
    y[temp] =sum6;
	temp += num_rows;
    y[temp] = sum7;
    temp += num_rows;
    y[temp] =sum8;
	temp += num_rows;
    y[temp] = sum9;
    temp += num_rows;
    y[temp] =sum10;
	temp += num_rows;
	y[temp] = sum11;
    temp += num_rows;
    y[temp] = sum12;
    temp += num_rows;
    y[temp] = sum13;
    temp += num_rows;
    y[temp] =sum14;
	temp += num_rows;
    y[temp] =sum15;
	temp += num_rows;
    y[temp] =sum16;

}
template <typename IndexType, typename ValueType>
void spmm_ell_device(const ell_matrix<IndexType,ValueType>& d_ell, 
                     const ValueType * d_x, 
                           ValueType * d_y)
{
    const unsigned int BLOCK_SIZE = 256;
    const dim3 grid = make_large_grid(d_ell.num_rows, BLOCK_SIZE);
    for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK){
		spmm_ell_kernel2<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
	}
}

template <typename IndexType, typename ValueType>
void spmm_ell_tex_device(const ell_matrix<IndexType,ValueType>& d_ell, 
                         const ValueType * d_x, 
                               ValueType * d_y)
{
    const unsigned int BLOCK_SIZE = 256;
    const dim3 grid = make_large_grid(d_ell.num_rows, BLOCK_SIZE);
  
    bind_x(d_x);
	for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK){
		spmm_ell_kernel2<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
	}
    unbind_x(d_x);
}


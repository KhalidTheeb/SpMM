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







template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmm_ell_kernel32(const IndexType num_rows, 
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
    temp += num_rows;
    ValueType sum17 = y[temp];
    temp += num_rows;
    ValueType sum18 = y[temp];
    temp += num_rows;
    ValueType sum19 = y[temp];
    temp += num_rows;
    ValueType sum20 = y[temp];
    temp += num_rows;
    ValueType sum21 = y[temp];
    temp += num_rows;
    ValueType sum22 = y[temp];
    temp += num_rows;
    ValueType sum23 = y[temp];
    temp += num_rows;
    ValueType sum24 = y[temp];
    temp += num_rows;
    ValueType sum25 = y[temp];
    temp += num_rows;
    ValueType sum26 = y[temp];
    temp += num_rows;
    ValueType sum27 = y[temp];
    temp += num_rows;
    ValueType sum28 = y[temp];
    temp += num_rows;
    ValueType sum29 = y[temp];
    temp += num_rows;
    ValueType sum30 = y[temp];	
    temp += num_rows;
    ValueType sum31 = y[temp];
    temp += num_rows;
    ValueType sum32 = y[temp];

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
            temp += num_cols;
            sum17 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum18 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum19 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum20 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum21 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum22 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum23 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum24 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum25 += A_ij * fetch_x<UseCache>(temp, x);	
            temp += num_cols;
            sum26 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum27 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum28 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum29 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum30 += A_ij * fetch_x<UseCache>(temp, x);
            temp += num_cols;
            sum31 += A_ij * fetch_x<UseCache>(temp, x);	
            temp += num_cols;
            sum32 += A_ij * fetch_x<UseCache>(temp, x);



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
    temp += num_rows;
    y[temp] = sum17;
    temp += num_rows;
    y[temp] = sum18;
    temp += num_rows;
    y[temp] =sum19;
    temp += num_rows;
    y[temp] =sum20;
    temp += num_rows;
    y[temp] =sum21;
    temp += num_rows;
    y[temp] = sum22;
    temp += num_rows;
    y[temp] =sum23;
    temp += num_rows;
    y[temp] = sum24;
    temp += num_rows;
    y[temp] =sum25;
    temp += num_rows;
    y[temp] = sum26;
    temp += num_rows;
    y[temp] = sum27;
    temp += num_rows;
    y[temp] = sum28;
    temp += num_rows;
    y[temp] =sum29;
    temp += num_rows;
    y[temp] =sum30;
    temp += num_rows;
    y[temp] =sum31;
    temp += num_rows;
    y[temp] =sum32;
}





template <typename IndexType, typename ValueType>
void spmm_ell_device(const ell_matrix<IndexType,ValueType>& d_ell, 
                     const ValueType * d_x, 
                           ValueType * d_y,
                           IndexType NUMVECTORS,
                           IndexType VECBLOCK)
{
    const unsigned int BLOCK_SIZE = 256;
    const dim3 grid = make_large_grid(d_ell.num_rows, BLOCK_SIZE);


     switch (NUMVECTORS){
	case 2:
        for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK)
        spmm_ell_kernel2<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
        break;
	
	
	case 4:
        for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK)
        spmm_ell_kernel4<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
        break;

	case 8:
        for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK)
        spmm_ell_kernel8<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
        break;

	case 16:
        for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK)
        spmm_ell_kernel16<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
        break;


	case 32:
        for (unsigned int vec=0; vec< NUMVECTORS; vec+=VECBLOCK)
        spmm_ell_kernel32<IndexType,ValueType,false> <<<grid, BLOCK_SIZE>>>
        (d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
        d_ell.Aj, d_ell.Ax,
        d_x+vec*d_ell.num_cols, d_y+vec*d_ell.num_rows);
        break;

    }


   

}

template <typename IndexType, typename ValueType>
void spmm_ell_tex_device(const ell_matrix<IndexType,ValueType>& d_ell, 
                         const ValueType * d_x, 
                               ValueType * d_y,
                               IndexType NUMVECTORS,
                               IndexType VECBLOCK)
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


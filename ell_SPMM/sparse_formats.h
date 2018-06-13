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

#include "mem.h"

////////////////////////////////////////////////////////////////////////////////
//! Defines the following sparse matrix formats
// ELL - ELLPACK/ITPACK
// CSR - Compressed Sparse Row
// CSC - Compressed Sparse Column
// COO - Coordinate
////////////////////////////////////////////////////////////////////////////////

template<typename IndexType>
struct matrix_shape
{
    typedef IndexType index_type;
    IndexType num_rows, num_cols, num_nonzeros;
};

// ELLPACK/ITPACK matrix format
template <typename IndexType, typename ValueType>
struct ell_matrix : public matrix_shape<IndexType> 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType stride;
    IndexType num_cols_per_row;

    IndexType * Aj;           //column indices stored in a (cols_per_row x stride) matrix
    ValueType * Ax;           //nonzero values stored in a (cols_per_row x stride) matrix
};

/*
 *  Compressed Sparse Row matrix (aka CRS)
 */
template <typename IndexType, typename ValueType>
struct csr_matrix : public matrix_shape<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType * Ap;  //row pointer
    IndexType * Aj;  //column indices
    ValueType * Ax;  //nonzeros
};

// COOrdinate matrix (aka IJV or Triplet format)
template <typename IndexType, typename ValueType>
struct coo_matrix : public matrix_shape<IndexType> 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType * I;  //row indices
    IndexType * J;  //column indices
    ValueType * V;  //nonzero values
};

/*
 *  Hybrid ELL/COO format
 */
template <typename IndexType, typename ValueType>
struct hyb_matrix : public matrix_shape<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    ell_matrix<IndexType,ValueType> ell; //ELL portion
    coo_matrix<IndexType,ValueType> coo; //COO portion
};


////////////////////////////////////////////////////////////////////////////////
//! sparse matrix memory management 
////////////////////////////////////////////////////////////////////////////////

template <typename IndexType, typename ValueType>
void delete_ell_matrix(ell_matrix<IndexType,ValueType>& ell, const memory_location loc){
    delete_array(ell.Aj, loc);  delete_array(ell.Ax, loc);
}

template <typename IndexType, typename ValueType>
void delete_csr_matrix(csr_matrix<IndexType,ValueType>& csr, const memory_location loc){
    delete_array(csr.Ap, loc);  delete_array(csr.Aj, loc);   delete_array(csr.Ax, loc);
}

template <typename IndexType, typename ValueType>
void delete_coo_matrix(coo_matrix<IndexType,ValueType>& coo, const memory_location loc){
    delete_array(coo.I, loc);   delete_array(coo.J, loc);   delete_array(coo.V, loc);
}

template <typename IndexType, typename ValueType>
void delete_hyb_matrix(hyb_matrix<IndexType,ValueType>& hyb, const memory_location loc){
    delete_ell_matrix(hyb.ell, loc);
    delete_coo_matrix(hyb.coo, loc);
}
////////////////////////////////////////////////////////////////////////////////
//! host functions
////////////////////////////////////////////////////////////////////////////////

template <typename IndexType, typename ValueType>
void delete_host_matrix(ell_matrix<IndexType,ValueType>& ell){ delete_ell_matrix(ell, HOST_MEMORY); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(csr_matrix<IndexType,ValueType>& csr){ delete_csr_matrix(csr, HOST_MEMORY); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(coo_matrix<IndexType,ValueType>& coo){ delete_coo_matrix(coo, HOST_MEMORY); }

template <class IndexType, class ValueType>
void delete_host_matrix(hyb_matrix<IndexType,ValueType>& hyb){  delete_hyb_matrix(hyb, HOST_MEMORY); }

////////////////////////////////////////////////////////////////////////////////
//! device functions
////////////////////////////////////////////////////////////////////////////////
template <typename IndexType, typename ValueType>
void delete_device_matrix(ell_matrix<IndexType,ValueType>& ell){ delete_ell_matrix(ell, DEVICE_MEMORY); }

template <typename IndexType, typename ValueType>
void delete_device_matrix(csr_matrix<IndexType,ValueType>& csr){ delete_csr_matrix(csr, DEVICE_MEMORY); }

template <class IndexType, class ValueType>
void delete_device_matrix(hyb_matrix<IndexType,ValueType>& hyb){  delete_hyb_matrix(hyb, DEVICE_MEMORY); }

////////////////////////////////////////////////////////////////////////////////
//! copy to device
////////////////////////////////////////////////////////////////////////////////

template <typename IndexType, typename ValueType>
ell_matrix<IndexType, ValueType> copy_matrix_to_device(const ell_matrix<IndexType, ValueType>& h_ell)
{
    ell_matrix<IndexType, ValueType> d_ell = h_ell; //copy fields
    d_ell.Aj = copy_array_to_device(h_ell.Aj, h_ell.stride * h_ell.num_cols_per_row);
    d_ell.Ax = copy_array_to_device(h_ell.Ax, h_ell.stride * h_ell.num_cols_per_row);
    return d_ell;
}


template <typename IndexType, typename ValueType>
csr_matrix<IndexType, ValueType> copy_matrix_to_device(const csr_matrix<IndexType, ValueType>& h_csr)
{
    csr_matrix<IndexType, ValueType> d_csr = h_csr; //copy fields
    d_csr.Ap = copy_array_to_device(h_csr.Ap, h_csr.num_rows + 1);
    d_csr.Aj = copy_array_to_device(h_csr.Aj, h_csr.num_nonzeros);
    d_csr.Ax = copy_array_to_device(h_csr.Ax, h_csr.num_nonzeros);
    return d_csr;
}

template <typename IndexType, typename ValueType>
hyb_matrix<IndexType, ValueType> copy_matrix_to_device(const hyb_matrix<IndexType, ValueType>& h_hyb)
{

    hyb_matrix<IndexType, ValueType> d_hyb = h_hyb; //copy fields
    d_hyb.ell = copy_matrix_to_device(h_hyb.ell);
    d_hyb.coo = copy_matrix_to_device(h_hyb.coo);
    return d_hyb;
}

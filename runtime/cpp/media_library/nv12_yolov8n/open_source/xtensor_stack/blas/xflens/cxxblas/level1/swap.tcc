/*
 *   Copyright (c) 2009, Michael Lehn
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CXXBLAS_LEVEL1_SWAP_TCC
#define CXXBLAS_LEVEL1_SWAP_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename X, typename Y>
void
swap_generic(IndexType n, X *x, IndexType incX, Y *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("swap_generic");

    for (IndexType i=0; i<n; ++i, x+=incX, y+=incY) {
        std::swap(*x, *y);
    }
}

template <typename IndexType, typename X, typename Y>
void
swap(IndexType n, X *x, IndexType incX, Y *y, IndexType incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    swap_generic(n, x, incX, y, incY);
}


#ifdef HAVE_CBLAS
// sswap
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n,
     float *x, IndexType incX,
     float *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sswap");

    cblas_sswap(n, x, incX, y, incY);
}

// dswap
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n,
     double *x, IndexType incX,
     double *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dswap");

    cblas_dswap(n, x, incX, y, incY);
}

// cswap
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n,
     ComplexFloat *x, IndexType incX,
     ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cswap");

    cblas_cswap(n, reinterpret_cast<float *>(x), incX,
                   reinterpret_cast<float *>(y), incY);
}

// zswap
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n,
     ComplexDouble *x, IndexType incX,
     ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zswap");

    cblas_zswap(n, reinterpret_cast<double *>(x), incX,
                   reinterpret_cast<double *>(y), incY);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_SWAP_TCC

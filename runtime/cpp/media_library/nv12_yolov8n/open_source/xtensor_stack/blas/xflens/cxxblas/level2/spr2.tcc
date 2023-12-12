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

#ifndef CXXBLAS_LEVEL2_SPR2_TCC
#define CXXBLAS_LEVEL2_SPR2_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
spr2_generic(StorageOrder order, StorageUpLo upLo,
             IndexType n,
             const ALPHA &alpha,
             const VX *x, IndexType incX,
             const VY *y, IndexType incY,
             MA *A)
{
    if (order==ColMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        spr2_generic(RowMajor, upLo, n, alpha, x, incX, y, incY, A);
        return;
    }
    #ifdef CXXBLAS_USE_XERBLA
        // insert error check here
    #endif
    if (upLo==Upper) {
        for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
            axpy_generic(n-i, alpha*x[iX], y+iY, incY,
                         A+i*(2*n-i+1)/2, IndexType(1));
            axpy_generic(n-i, alpha*y[iY], x+iX, incX,
                         A+i*(2*n-i+1)/2, IndexType(1));
        }
    } else {
        for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
            axpy_generic(i+1, alpha*x[iX], y, incY,
                         A+i*(i+1)/2, IndexType(1));
            axpy_generic(i+1, alpha*y[iY], x, incX,
                         A+i*(i+1)/2, IndexType(1));
        }
    }
}

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
spr2(StorageOrder order, StorageUpLo upLo,
     IndexType n,
     const ALPHA &alpha,
     const VX *x, IndexType incX,
     const VY *y, IndexType incY,
     MA *A)
{
    CXXBLAS_DEBUG_OUT("spr2_generic");

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    spr2_generic(order, upLo, n, alpha, x, incX, y, incY, A);
}

#ifdef HAVE_CBLAS

// sspr2
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
spr2(StorageOrder order,   StorageUpLo upLo,
      IndexType n,
      float alpha,
      const float *x, IndexType incX,
      const float *y, IndexType incY,
      float *A)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sspr2");

    cblas_sspr2(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
               n,
               alpha,
               reinterpret_cast<const float *>(x), incX,
               reinterpret_cast<const float *>(y), incY,
               reinterpret_cast<float *>(A));
}

// dspr2
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
spr2(StorageOrder order,   StorageUpLo upLo,
      IndexType n,
      double alpha,
      const double *x, IndexType incX,
      const double *y, IndexType incY,
      double *A)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dspr2");

    cblas_dspr2(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                n,
                alpha,
                reinterpret_cast<const double *>(x), incX,
                reinterpret_cast<const double *>(y), incY,
                reinterpret_cast<double *>(A));
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_SPR2_TCC

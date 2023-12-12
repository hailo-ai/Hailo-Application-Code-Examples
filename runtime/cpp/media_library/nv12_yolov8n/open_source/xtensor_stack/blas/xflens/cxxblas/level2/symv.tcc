/*
 *   Copyright (c) 2010, Michael Lehn
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

#ifndef CXXBLAS_LEVEL2_SYMV_TCC
#define CXXBLAS_LEVEL2_SYMV_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

#ifdef USE_CXXLAPACK
#include "xflens/cxxlapack/cxxlapack.h"
#endif

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
symv_generic(StorageOrder order, StorageUpLo upLo,
             IndexType n,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const VX *x, IndexType incX,
             const BETA &beta,
             VY *y, IndexType incY)
{
    if (order==ColMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
    }
    scal_init_generic(n, beta, y, incY);
    if (upLo==Upper) {
        for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
            VY y_ = VY(0);
            dotu_generic(n-i, A+i*ldA+i, IndexType(1), x+i*incX, incX, y_);
            y[iY] += alpha*y_;
        }
        for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
            VY y_ = VY(0);
            dotu_generic(i, A+i, ldA, x, incX, y_);
            y[iY] += alpha*y_;
        }
    } else {
        for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
            VY y_ = VY(0);
            dotu_generic(i, A+i*ldA, IndexType(1), x, incX, y_);
            y[iY] += alpha*y_;
        }
        for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
            VY y_ = VY(0);
            dotu_generic(n-i, A+i*ldA+i, ldA, x+i*incX, incX, y_);
            y[iY] += alpha*y_;
        }
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
symv(StorageOrder order, StorageUpLo upLo,
     IndexType n,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const VX *x, IndexType incX,
     const BETA &beta,
     VY *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("symv_generic");

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    symv_generic(order, upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
}

#ifdef HAVE_CBLAS

// ssymv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symv(StorageOrder order, StorageUpLo upLo,
     IndexType n, float alpha,
     const float *A, IndexType ldA,
     const float *x, IndexType incX,
     float beta,
     float *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ssymv");

    cblas_ssymv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                n, alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
}

// dsymv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symv(StorageOrder order, StorageUpLo upLo,
     IndexType n, double alpha,
     const double *A, IndexType ldA,
     const double *x, IndexType incX,
     double beta,
     double *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dsymv");

    cblas_dsymv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                n, alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
}

#ifdef USE_CXXLAPACK

// csymv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symv(StorageOrder order, StorageUpLo upLo,
     IndexType n,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat *x, IndexType incX,
     const ComplexFloat &beta,
     ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_symv");

    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
    }

    cxxlapack::symv(getF77BlasChar(upLo),
                    n, &alpha,
                    A, ldA,
                    x, incX,
                    beta,
                    y, incY);;
}

// zsymv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symv(StorageOrder order, StorageUpLo upLo,
     IndexType n,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble &beta,
     ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_symv");

    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
    }
    cxxlapack::symv(getF77BlasChar(upLo),
                    n, alpha,
                    A, ldA,
                    x, incX,
                    beta,
                    y, incY);;
}

#endif // USE_CXXLAPACK

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_SYMV_TCC

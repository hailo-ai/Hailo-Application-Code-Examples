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

#ifndef CXXBLAS_LEVEL2_GER_TCC
#define CXXBLAS_LEVEL2_GER_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
geru_generic(StorageOrder order,
             IndexType m, IndexType n,
             const ALPHA &alpha,
             const VX *x, IndexType incX,
             const VY *y, IndexType incY,
             MA *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("geru_generic");
    if (order==ColMajor) {
        geru_generic(RowMajor, n, m,
                     alpha, y, incY, x, incX,
                     A, ldA);
        return;
    }
    #ifdef CXXBLAS_USE_XERBLA
        // insert error check here
    #endif
    for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
        axpy_generic(n, alpha*x[iX], y, incY, A+i*ldA, IndexType(1));
    }
}

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
gerc_generic(StorageOrder order, Transpose conjugateA,
             IndexType m, IndexType n,
             const ALPHA &alpha,
             const VX *x, IndexType incX,
             const VY *y, IndexType incY,
             MA *A, IndexType ldA)
{
    if (order==ColMajor) {
        conjugateA = Transpose(conjugateA^Conj);
        gerc_generic(RowMajor, conjugateA, n, m,
                     alpha, y, incY, x, incX,
                     A, ldA);
        return;
    }

    #ifdef CXXBLAS_USE_XERBLA
        // insert error check here
    #endif
    if (conjugateA==Conj) {
        for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
            axpy_generic(n, alpha*conjugate(x[iX]),
                            y, incY,
                            A+i*ldA, IndexType(1));
        }
    } else {
        for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
            acxpy_generic(n, alpha*x[iX], y, incY, A+i*ldA, IndexType(1));
        }
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
ger(StorageOrder order,
    IndexType m, IndexType n,
    const ALPHA &alpha,
    const VX *x, IndexType incX,
    const VY *y, IndexType incY,
    MA *A, IndexType ldA)
{
    geru(order, m, n, alpha, x, incX, y, incY, A, ldA);
}

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
geru(StorageOrder order,
     IndexType m, IndexType n,
     const ALPHA &alpha,
     const VX *x, IndexType incX,
     const VY *y, IndexType incY,
     MA *A, IndexType ldA)
{
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    geru_generic(order, m, n, alpha, x, incX, y, incY, A, ldA);
}

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
void
gerc(StorageOrder order,
     IndexType m, IndexType n,
     const ALPHA &alpha,
     const VX *x, IndexType incX,
     const VY *y, IndexType incY,
     MA *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("gerc_generic");

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    gerc_generic(order, NoTrans, m, n, alpha, x, incX, y, incY, A, ldA);
}

#ifdef HAVE_CBLAS

// sger
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    const float &alpha,
    const float *x, IndexType incX,
    const float *y, IndexType incY,
    float *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sger");

    cblas_sger(CBLAS::getCblasType(order),
               m, n,
               alpha,
               x, incX,
               y, incY,
               A, ldA);
}

// dger
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    const double &alpha,
    const double *x, IndexType incX,
    const double *y, IndexType incY,
    double *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dger");

    cblas_dger(CBLAS::getCblasType(order),
               m, n,
               alpha,
               x, incX,
               y, incY,
               A, ldA);
}

// cgeru
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
geru(StorageOrder order,
     IndexType m, IndexType n,
     const ComplexFloat &alpha,
     const ComplexFloat *x, IndexType incX,
     const ComplexFloat *y, IndexType incY,
     ComplexFloat *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cgeru");

    cblas_cgeru(CBLAS::getCblasType(order),
                m, n,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(x), incX,
                reinterpret_cast<const float *>(y), incY,
                reinterpret_cast<float *>(A), ldA);
}

// zgeru
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
geru(StorageOrder order,
     IndexType m, IndexType n,
     const ComplexDouble &alpha,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble *y, IndexType incY,
     ComplexDouble *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zgeru");

    cblas_zgeru(CBLAS::getCblasType(order),
                m, n,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(x), incX,
                reinterpret_cast<const double *>(y), incY,
                reinterpret_cast<double *>(A), ldA);
}

// cgerc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gerc(StorageOrder order,
     IndexType m, IndexType n,
     const ComplexFloat &alpha,
     const ComplexFloat *x, IndexType incX,
     const ComplexFloat *y, IndexType incY,
     ComplexFloat *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cgerc");

    cblas_cgerc(CBLAS::getCblasType(order),
                m, n,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(x), incX,
                reinterpret_cast<const float *>(y), incY,
                reinterpret_cast<float *>(A), ldA);
}

// zgerc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gerc(StorageOrder order,
     IndexType m, IndexType n,
     const ComplexDouble &alpha,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble *y, IndexType incY,
     ComplexDouble *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zgerc");

    cblas_zgerc(CBLAS::getCblasType(order),
                m, n,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(x), incX,
                reinterpret_cast<const double *>(y), incY,
                reinterpret_cast<double *>(A), ldA);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_GER_TCC

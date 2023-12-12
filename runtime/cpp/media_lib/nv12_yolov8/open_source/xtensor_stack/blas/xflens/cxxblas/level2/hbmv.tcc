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

#ifndef CXXBLAS_LEVEL2_HBMV_TCC
#define CXXBLAS_LEVEL2_HBMV_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
hbmv_generic(StorageOrder order, StorageUpLo upLo, Transpose conjugateA,
             IndexType n, IndexType k,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const VX *x, IndexType incX,
             const BETA &beta,
             VY *y, IndexType incY)
{
    using std::max;
    using std::min;

    if (order==ColMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        conjugateA = Transpose(conjugateA^Conj);
    }
    scal_init_generic(n, beta, y, incY);
    if (upLo==Upper) {
        if (conjugateA==Conj) {
            for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
                IndexType len = min(k+1, n-i);

                y[iY] += alpha*cxxblas::real(A[ldA*i])*x[iX];

                VY y_;
                dot_generic(len-1, A+ldA*i+1, IndexType(1),
                                   x+iX+incX, IndexType(incX),
                                   y_);
                y[iY] += alpha*y_;

                axpy_generic(len-1, x[iX] * alpha,
                                    A+ldA*i+1, IndexType(1),
                                    y+iY+incY, incY);
            }
        } else {
            for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
                IndexType len = min(k+1, n-i);

                y[iY] += alpha*cxxblas::real(A[ldA*i])*x[iX];

                VY y_;
                dotu_generic(len-1, A+ldA*i+1, IndexType(1),
                                    x+iX+incX, IndexType(incX),
                                    y_);
                y[iY] += alpha*y_;

                acxpy_generic(len-1, x[iX] * alpha,
                                     A+ldA*i+1, IndexType(1),
                                     y+iY+incY, incY);
            }
        }
    } else { /* upLo==Lower */
        if (conjugateA==Conj) {
            for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
                IndexType iA = max(k-i, IndexType(0));
                IndexType len = min(k, i) + 1;
                IndexType i_ = max(i-k, IndexType(0));

                y[iY] += alpha*cxxblas::real(A[ldA*i+iA+ len-1])*x[i*incX];

                VY y_;
                dot_generic(len-1, A+ldA*i+iA, IndexType(1),
                                   x+i_*incX, IndexType(incX),
                                   y_);
                y[iY] += alpha*y_;

                axpy_generic(len-1, x[i*incX] * alpha,
                                    A+ldA*i+iA, IndexType(1),
                                    y+i_*incY, incY);
            }
        } else {
            for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
                IndexType iA = max(k-i, IndexType(0));
                IndexType len = min(k, i) + 1;
                IndexType i_ = max(i-k, IndexType(0));

                y[iY] += alpha*cxxblas::real(A[ldA*i+iA+ len-1])*x[i*incX];

                VY y_;
                dotu_generic(len-1, A+ldA*i+iA, IndexType(1),
                                    x+i_*incX, IndexType(incX),
                                    y_);
                y[iY] += alpha*y_;

                acxpy_generic(len-1, x[i*incX] * alpha,
                                     A+ldA*i+iA, IndexType(1),
                                     y+i_*incY, incY);
            }
        }
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
hbmv(StorageOrder order, StorageUpLo upLo,
     IndexType n, IndexType k,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const VX *x, IndexType incX,
     const BETA &beta,
     VY *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("hbmv_generic");

    if (n==0) {
        return;
    }
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    hbmv_generic(order, upLo, NoTrans,
                 n, k,
                 alpha, A, ldA,
                 x, incX,
                 beta,
                 y, incY);
}


#ifdef HAVE_CBLAS

// chbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
hbmv(StorageOrder order, Transpose trans,
     IndexType n, IndexType k,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat *x, IndexType incX,
     const ComplexFloat &beta,
     ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_chbmv");

    cblas_chbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                n, k,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<const float *>(x), incX,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(y), incY);
}

// zhbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
hbmv(StorageOrder order, Transpose trans,
     IndexType n, IndexType k,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble &beta,
     ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zhbmv");

    cblas_zhbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                n, k,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<const double *>(x), incX,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(y), incY);
}

#endif // HAVE_CBLAS


} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_HBMV_TCC

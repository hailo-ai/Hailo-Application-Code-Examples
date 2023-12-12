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

#ifndef CXXBLAS_LEVEL2_GEMV_TCC
#define CXXBLAS_LEVEL2_GEMV_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
gemv_generic(StorageOrder order, Transpose transA, Transpose conjX,
             IndexType m, IndexType n,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const VX *x, IndexType incX,
             const BETA &beta,
             VY *y, IndexType incY)
{
    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        gemv_generic(RowMajor, transA, conjX, n, m, alpha, A, ldA,
                     x, incX, beta, y, incY);
        return;
    }
    if ((transA==NoTrans) || (transA==Conj)) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        scal_init_generic(m, beta, y, incY);
        if (conjX==NoTrans) {
            if (transA==Conj) {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    VY y_;
                    dot_generic(n, A+i*ldA, IndexType(1), x, incX, y_);
                    y[iY] += alpha * y_;
                }
            } else {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    VY y_;
                    dotu_generic(n, A+i*ldA, IndexType(1), x, incX, y_);
                    y[iY] += alpha * y_;
                }
            }
        } else if (conjX==Conj) {
            if (transA==Conj) {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    VY y_;
                    dotu_generic(n, A+i*ldA, IndexType(1), x, incX, y_);
                    y[iY] += alpha*cxxblas::conjugate(y_);
                }
            } else {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    VY y_;
                    dot_generic(n, x, incX, A+i*ldA, IndexType(1), y_);
                    y[iY] += alpha * y_;
                }
            }
        }
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }

        scal_init_generic(n, beta, y, incY);
        if (conjX==NoTrans) {
            if (transA==ConjTrans) {
                for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
                    VY y_;
                    dot_generic(m, A+i, ldA, x, incX, y_);
                    y[iY] += alpha * y_;
                }
            } else {
                for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
                    VY y_;
                    dotu_generic(m, A+i, ldA, x, incX, y_);
                    y[iY] += alpha * y_;
                }
            }
        } else if (conjX==Conj) {
            if (transA==ConjTrans) {
                for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
                    VY y_;
                    dotu_generic(m, A+i, ldA, x, incX, y_);
                    y[iY] += alpha*cxxblas::conjugate(y_);
                }
            } else {
                for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
                    VY y_;
                    dot_generic(m, x, incX, A+i, ldA, y_);
                    y[iY] += alpha * y_;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const VX *x, IndexType incX,
     const BETA &beta,
     VY *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv_generic");

    if ((m==0) || (n==0)) {
        return;
    }
    gemv_generic(order, trans, NoTrans, m, n,
                 alpha, A, ldA, x, incX,
                 beta, y, incY);
}

#ifdef HAVE_CBLAS

// sgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     float alpha,
     const float *A, IndexType ldA,
     const float *x, IndexType incX,
     float beta,
     float *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sgemv");

    cblas_sgemv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
}

// dgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     double alpha,
     const double *A, IndexType ldA,
     const double *x, IndexType incX,
     double beta,
     double *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dgemv");

    cblas_dgemv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
}

// cgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat *x, IndexType incX,
     const ComplexFloat &beta,
     ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cgemv");

    if (trans==Conj) {
        order  = (order==RowMajor) ? ColMajor : RowMajor;
        gemv(order, ConjTrans, n, m, alpha, A, ldA,
                     x, incX, beta, y, incY);
        return;
    }

#   ifdef CXXBLAS_NO_TEMPORARY
    if (order==RowMajor && trans==ConjTrans) {
        CXXBLAS_DEBUG_OUT("gemv_generic");
        gemv_generic(order, trans, NoTrans, m, n,
                     alpha, A, ldA, x, incX,
                     beta, y, incY);
        return;
    }
#   endif

    cblas_cgemv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<const float *>(x), incX,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(y), incY);
}

// zgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble &beta,
     ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zgemv");

    if (trans==Conj) {
        order  = (order==RowMajor) ? ColMajor : RowMajor;
        gemv(order, ConjTrans, n, m, alpha, A, ldA,
                     x, incX, beta, y, incY);
        return;
    }

#   ifdef CXXBLAS_NO_TEMPORARY
    if (order==RowMajor && trans==ConjTrans) {
        CXXBLAS_DEBUG_OUT("gemv_generic");
        gemv_generic(order, trans, NoTrans, m, n,
                     alpha, A, ldA, x, incX,
                     beta, y, incY);
        return;
    }
#   endif

    cblas_zgemv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<const double *>(x), incX,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(y), incY);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_GEMV_TCC

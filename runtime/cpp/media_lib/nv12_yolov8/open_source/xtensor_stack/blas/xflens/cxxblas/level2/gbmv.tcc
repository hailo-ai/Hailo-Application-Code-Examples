/*
 *   Copyright (c) 2012, Klaus Pototzky
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

#ifndef CXXBLAS_LEVEL2_GBMV_TCC
#define CXXBLAS_LEVEL2_GBMV_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
gbmv_generic(StorageOrder order, Transpose trans, Transpose conjX,
             IndexType m, IndexType n,
             IndexType kl, IndexType ku,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const VX *x, IndexType incX,
             const BETA &beta,
             VY *y, IndexType incY)
{
    if (order==ColMajor) {
        trans = Transpose(trans^Trans);
        gbmv_generic(RowMajor, trans, conjX, n, m, ku, kl, alpha, A, ldA,
                     x, incX, beta, y, incY);
        return;
    }
    if ((trans==NoTrans) || (trans==Conj)) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        scal_init_generic(m, beta, y, incY);
        if (trans==NoTrans) {
            if (conjX==NoTrans) {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iX  = std::max(i-kl, IndexType(0))*incX;

                    VY y_;
                    dotu_generic(len, A+ldA*i+iA, IndexType(1),
                                 x+iX, IndexType(incX), y_);
                    y[iY] += alpha * y_;
                }
            } else if (conjX==Conj) {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iX  = std::max(i-kl, IndexType(0))*incX;

                    VY y_;
                    dot_generic(len, x+iX, IndexType(incX),
                                A+ldA*i+iA, IndexType(1), y_);
                    y[iY] += alpha * y_;
                }
            }
        } else {
            if (conjX==NoTrans) {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iX  = std::max(i-kl, IndexType(0))*incX;

                    VY y_;
                    dot_generic(len, A+ldA*i+iA, IndexType(1),
                                x+iX, IndexType(incX),
                                y_);
                    y[iY] += alpha * y_;
                }
            } else if (conjX==Conj) {
                for (IndexType i=0, iY=0; i<m; ++i, iY+=incY) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iX  = std::max(i-kl, IndexType(0))*incX;

                    VY y_;
                    dotu_generic(len, A+ldA*i+iA, IndexType(1),
                                 x+iX, IndexType(incX),
                                 y_);
                    y[iY] += alpha*conjugate(y_);
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
        if (trans==Trans) {
            if (conjX == NoTrans) {
                for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iY  = std::max(i - kl, IndexType(0))*incY;

                    axpy_generic(len, x[iX] * alpha,
                                 A+ldA*i+iA, IndexType(1),
                                 y+iY, incY);
                }
            } else if (conjX==Conj) {
                for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iY  = std::max(i - kl, IndexType(0))*incY;

                    VY x_ = conjugate(x[iX]);
                    axpy_generic(len, x_ * alpha,
                                 A+ldA*i+iA, IndexType(1),
                                 y+iY, incY);
                }
            }
        } else {
            if (conjX == NoTrans) {
                for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iY  = std::max(i - kl, IndexType(0))*incY;

                    acxpy_generic(len, x[iX] * alpha,
                                  A+ldA*i+iA, IndexType(1),
                                  y+iY, incY);
                }
            } else if (conjX==Conj) {
                for (IndexType i=0, iX=0; i<m; ++i, iX+=incX) {
                    IndexType iA  = std::max(kl-i, IndexType(0));
                    IndexType len = std::min(kl+ku+1, kl-i+n) - iA;
                    IndexType iY  = std::max(i - kl, IndexType(0))*incY;

                    VY x_ = conjugate(x[iX]);
                    acxpy_generic(len, x_ * alpha,
                                  A+ldA*i+iA, IndexType(1),
                                  y+iY, incY);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
gbmv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     IndexType kl, IndexType ku,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const VX *x, IndexType incX,
     const BETA &beta,
     VY *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gbmv_generic");

    if ((m==0) || (n==0)) {
        return;
    }
    gbmv_generic(order, trans, NoTrans, m, n, kl, ku, alpha, A, ldA,
                 x, incX, beta, y, incY);
}



#ifdef HAVE_CBLAS

// sgbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gbmv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     IndexType kl, IndexType ku,
     float alpha,
     const float *A, IndexType ldA,
     const float *x, IndexType incX,
     float beta,
     float *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sgbmv");

    cblas_sgbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n, kl, ku,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
}

// dgbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gbmv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     IndexType kl, IndexType ku,
     double alpha,
     const double *A, IndexType ldA,
     const double *x, IndexType incX,
     double beta,
     double *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dgbmv");

    cblas_dgbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n, kl, ku,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
}

// cgbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gbmv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     IndexType kl, IndexType ku,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat *x, IndexType incX,
     const ComplexFloat &beta,
     ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cgbmv");

    if (trans==Conj) {
        order  = (order==RowMajor) ? ColMajor : RowMajor;
        gbmv(order, ConjTrans, n, m, ku, kl, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

#   ifdef CXXBLAS_NO_TEMPORARY
        if (order==RowMajor && trans==ConjTrans) {
            CXXBLAS_DEBUG_OUT("gbmv_generic");
            gbmv_generic(order, trans, NoTrans, m, n, kl, ku, alpha, A, ldA,
                         x, incX, beta, y, incY);
            return;
        }
#   endif

    cblas_cgbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n, kl, ku,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<const float *>(x), incX,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(y), incY);
}

// zgbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gbmv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     IndexType kl, IndexType ku,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble &beta,
     ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zgbmv");

    if (trans==Conj) {
        order  = (order==RowMajor) ? ColMajor : RowMajor;
        gbmv(order, ConjTrans, n, m, ku, kl, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

#   ifdef CXXBLAS_NO_TEMPORARY
        if (order==RowMajor && trans==ConjTrans) {
            CXXBLAS_DEBUG_OUT("gbmv_generic");
            gbmv_generic(order, trans, NoTrans, m, n, kl, ku, alpha, A, ldA,
                         x, incX, beta, y, incY);
            return;
        }
#   endif

    cblas_zgbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(trans),
                m,  n, kl, ku,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<const double *>(x), incX,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(y), incY);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_GBMV_TCC

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

#ifndef CXXBLAS_LEVEL2_TPMV_TCC
#define CXXBLAS_LEVEL2_TPMV_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename MA, typename VX>
void
tpmv_generic(StorageOrder order, StorageUpLo upLo,
             Transpose transA, Diag diag,
             IndexType n,
             const MA *A,
             VX *x, IndexType incX)
{
    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tpmv_generic(RowMajor, upLo, transA, diag, n, A, x, incX);
        return;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

    if (transA==NoTrans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dotu_generic(n-i, A+i*(2*n-i+1)/2, IndexType(1),
                                      x+iX, incX, x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n-1; ++i, iX+=incX) {
                    VX x_;
                    dotu_generic(n-i-1, A+i*(2*n-i+1)/2+1, IndexType(1),
                                        x+iX+incX, incX, x_);
                    x[iX] += x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dotu_generic(i+1, A+i*(i+1)/2, IndexType(1),
                                      x, incX, x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>0; --i, iX-=incX) {
                    VX x_;
                    dotu_generic(i, A+i*(i+1)/2, IndexType(1),
                                    x, incX, x_);
                    x[iX] += x_;
                }
            }
        }
    }
    if (transA==Conj) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dot_generic(n-i, A+i*(2*n-i+1)/2, IndexType(1),
                                     x+iX, incX, x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n-1; ++i, iX+=incX) {
                    VX x_;
                    dot_generic(n-i-1, A+i*(2*n-i+1)/2+1, IndexType(1),
                                       x+iX+incX, incX, x_);
                    x[iX] += x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dot_generic(i+1, A+i*(i+1)/2, IndexType(1),
                                     x, incX, x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>0; --i, iX-=incX) {
                    VX x_;
                    dot_generic(i, A+i*(i+1)/2, IndexType(1),
                                   x, incX, x_);
                    x[iX] += x_;
                }
            }
        }
    }
    if (transA==Trans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    axpy_generic(n-1-i, x[iX], A+i*(2*n-i+1)/2+1, IndexType(1),
                                               x+iX+incX, incX);
                    x[iX] *= *(A+i*(2*n-i+1)/2);
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    axpy_generic(n-1-i, x[iX], A+i*(2*n-i+1)/2+1, IndexType(1),
                                               x+iX+incX, incX);
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    axpy_generic(i, x[iX], A+i*(i+1)/2, IndexType(1),
                                           x, incX);
                    x[iX] *= *(A+i*(i+3)/2);
                }
            } else {
                for (IndexType i=1, iX=i*incX; i<n; ++i, iX+=incX) {
                    axpy_generic(i, x[iX], A+i*(i+1)/2, IndexType(1),
                                           x, incX);
                }
            }
        }
    }
    if (transA==ConjTrans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    acxpy_generic(n-1-i, x[iX], A+i*(2*n-i+1)/2+1, IndexType(1),
                                                x+iX+incX, incX);
                    x[iX] *= conjugate(A[i*(2*n-i+1)/2]);
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    acxpy_generic(n-1-i, x[iX], A+i*(2*n-i+1)/2+1, IndexType(1),
                                                x+iX+incX, incX);
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    acxpy_generic(i, x[iX], A+i*(i+1)/2, IndexType(1),
                                            x, incX);
                    x[iX] *= conjugate(A[i*(i+3)/2]);
                }
            } else {
                for (IndexType i=1, iX=i*incX; i<n; ++i, iX+=incX) {
                    acxpy_generic(i, x[iX], A+i*(i+1)/2, IndexType(1),
                                            x, incX);
                }
            }
        }
    }
}

template <typename IndexType, typename MA, typename VX>
void
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const MA *A,
     VX *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("tpmv_generic");

    tpmv_generic(order, upLo, transA, diag, n, A, x, incX);
}

#ifdef HAVE_CBLAS

// stpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const float *A,
     float *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_stpmv");

    cblas_stpmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                A,
                x, incX);
}

// dtpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const double *A,
     double *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dtpmv");

    cblas_dtpmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                A,
                x, incX);
}

// ctpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const ComplexFloat *A,
     ComplexFloat *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ctpmv");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("tpmv_generic");
        tpmv_generic(order, upLo, transA, diag, n, A, x, incX);

        return;
    }

    cblas_ctpmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                reinterpret_cast<const float *>(A),
                reinterpret_cast<float *>(x), incX);
}

// ztpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const ComplexDouble *A,
     ComplexDouble *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ztpmv");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("tpmv_generic");
        tpmv_generic(order, upLo, transA, diag, n, A, x, incX);

        return;
    }

    cblas_ztpmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                reinterpret_cast<const double *>(A),
                reinterpret_cast<double *>(x), incX);
}
#endif // HAVE_CBLAS

} // namespace flens

#endif // CXXBLAS_LEVEL2_TPMV_TCC


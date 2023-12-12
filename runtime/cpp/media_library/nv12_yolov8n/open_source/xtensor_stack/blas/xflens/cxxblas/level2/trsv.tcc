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

#ifndef CXXBLAS_LEVEL2_TRSV_TCC
#define CXXBLAS_LEVEL2_TRSV_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename MA, typename VX>
void
trsv_generic(StorageOrder order, StorageUpLo upLo,
             Transpose transA, Diag diag,
             IndexType n,
             const MA *A, IndexType ldA,
             VX *x, IndexType incX)
{
    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Lower) ? Upper : Lower;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

    if (transA==NoTrans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dotu_generic(n-i-1, A+i*(ldA+1)+1, IndexType(1),
                                        x+iX+incX, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= A[i*(ldA+1)];
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dotu_generic(n-i-1, A+i*(ldA+1)+1, IndexType(1),
                                        x+iX+incX, incX, x_);
                    x[iX] -= x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dotu_generic(i, A+i*ldA, IndexType(1),
                                    x, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= A[i*(ldA+1)];
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dotu_generic(i, A+i*ldA, IndexType(1),
                                    x, incX, x_);
                    x[iX] -= x_;
                }
            }
        }
    }
    if (transA==Conj) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dot_generic(n-i-1, A+i*(ldA+1)+1, IndexType(1),
                                       x+iX+incX, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= conjugate(A[i*(ldA+1)]);
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dot_generic(n-i-1, A+i*(ldA+1)+1, IndexType(1),
                                       x+iX+incX, incX, x_);
                    x[iX] -= x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dot_generic(i, A+i*ldA, IndexType(1),
                                   x, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= conjugate(A[i*(ldA+1)]);
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dot_generic(i, A+i*ldA, IndexType(1),
                                   x, incX, x_);
                    x[iX] -= x_;
                }
            }
        }
    }
    if (transA==Trans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dotu_generic(i, A+i, IndexType(ldA),
                                    x, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= A[i*(ldA+1)];
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dotu_generic(i, A+i, IndexType(ldA),
                                    x, incX, x_);
                    x[iX] -= x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dotu_generic(n-i-1, A+i*(ldA+1)+ldA, IndexType(ldA),
                                        x+iX+incX, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= A[i*(ldA+1)];
                }
            } else {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dotu_generic(n-i-1, A+i*(ldA+1)+ldA, IndexType(ldA),
                                        x+iX+incX, incX, x_);
                    x[iX] -= x_;
                }
            }
        }
    }
    if (transA==ConjTrans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dot_generic(i, A+i, IndexType(ldA),
                                   x, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= conjugate(A[i*(ldA+1)]);
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    VX x_;
                    dot_generic(i, A+i, IndexType(ldA),
                                   x, incX, x_);
                    x[iX] -= x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dot_generic(n-i-1, A+i*(ldA+1)+ldA, IndexType(ldA),
                                       x+iX+incX, incX, x_);
                    x[iX] -= x_;
                    x[iX] /= conjugate(A[i*(ldA+1)]);
                }
            } else {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    VX x_;
                    dot_generic(n-i-1, A+i*(ldA+1)+ldA, IndexType(ldA),
                                       x+iX+incX, incX, x_);
                    x[iX] -= x_;
                }
            }
        }
    }
}

template <typename IndexType, typename MA, typename VX>
void
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const MA *A, IndexType ldA,
     VX *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("trsv_generic");

    trsv_generic(order, upLo, transA, diag, n, A, ldA, x, incX);
}

#ifdef HAVE_CBLAS

// strsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const float *A, IndexType ldA,
     float *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_strsv");

    cblas_strsv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                A, ldA,
                x, incX);
}

// dtrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const double *A, IndexType ldA,
     double *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dtrsv");

    cblas_dtrsv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                A, ldA,
                x, incX);
}

// ctrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const ComplexFloat *A, IndexType ldA,
     ComplexFloat *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ctrsv");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("trsv_generic");
        trsv_generic(order, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }

    cblas_ctrsv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<float *>(x), incX);
}

// ztrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const ComplexDouble *A, IndexType ldA,
     ComplexDouble *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ztrsv");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("trsv_generic");
        trsv_generic(order, upLo, transA, diag, n, A, ldA, x, incX);

        return;
    }

    cblas_ztrsv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n,
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<double *>(x), incX);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_TRSV_TCC


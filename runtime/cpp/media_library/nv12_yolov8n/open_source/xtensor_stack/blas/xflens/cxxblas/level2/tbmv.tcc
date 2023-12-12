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

#ifndef CXXBLAS_LEVEL2_TBMV_TCC
#define CXXBLAS_LEVEL2_TBMV_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename MA, typename VX>
void
tbmv_generic(StorageOrder order, StorageUpLo upLo,
             Transpose transA, Diag diag,
             IndexType n, IndexType k,
             const MA *A, IndexType ldA,
             VX *x, IndexType incX)
{
    using std::max;
    using std::min;

    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbmv_generic(RowMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

    if (transA==NoTrans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType len = min(k+1, n-i);

                    VX x_;
                    dotu_generic(len, A+ldA*i, IndexType(1),
                                      x+iX, IndexType(incX),
                                      x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType len = min(k+1, n-i);

                    VX x_;
                    dotu_generic(len-1, A+ldA*i+1, IndexType(1),
                                        x+iX+incX, IndexType(incX),
                                        x_);
                    x[iX] += x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType i_ = max(i-k, IndexType(0));

                    VX x_;
                    dotu_generic(len, A+ldA*i+iA, IndexType(1),
                                      x+i_*incX, IndexType(incX),
                                      x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType i_ = max(i-k, IndexType(0));

                    VX x_;
                    dotu_generic(len-1, A+ldA*i+iA, IndexType(1),
                                        x+i_*incX, IndexType(incX),
                                        x_);
                    x[iX] += x_;
                }
            }
        }
    }
    if (transA==Conj) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType len = min(k+1, n-i);

                    VX x_;
                    dot_generic(len, A+ldA*i, IndexType(1),
                                     x+iX, IndexType(incX),
                                     x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType len = min(k+1, n-i);

                    VX x_;
                    dot_generic(len-1, A+ldA*i+1, IndexType(1),
                                       x+iX+incX, IndexType(incX),
                                       x_);
                    x[iX] += x_;
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType i_ = max(i-k, IndexType(0));

                    VX x_;
                    dot_generic(len, A+ldA*i+iA, IndexType(1),
                                     x+i_*incX, IndexType(incX),
                                     x_);
                    x[iX] = x_;
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType i_ = max(i-k, IndexType(0));

                    VX x_;
                    dot_generic(len-1, A+ldA*i+iA, IndexType(1),
                                       x+i_*incX, IndexType(incX),
                                       x_);
                    x[iX] += x_;
                }
            }
        }
    }
    if (transA==Trans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType len = min(k+1, n-i);

                    axpy_generic(len-1, x[iX],
                                        A+ldA*i+1, IndexType(1),
                                        x+iX+incX, incX);
                    x[iX] *= *(A+ldA*i);
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType len = min(k+1, n-i);

                    axpy_generic(len-1, x[iX],
                                        A+ldA*i+1, IndexType(1),
                                        x+iX+incX, incX);
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType iY = max(i-k, IndexType(0))*incX;

                    axpy_generic(len-1, x[iX],
                                        A+ldA*i+iA, IndexType(1),
                                        x+iY, incX);
                    x[iX] *= *(A+ldA*i+iA+len-1);
                }
            } else {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType iY = max(i-k, IndexType(0))*incX;

                    axpy_generic(len-1, x[iX],
                                        A+ldA*i+iA, IndexType(1),
                                        x+iY, incX);
                }
            }
        }
    }
    if (transA==ConjTrans) {
        if (upLo==Upper) {
            if (diag==NonUnit) {
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType len = min(k+1, n-i);

                    acxpy_generic(len-1, x[iX],
                                         A+ldA*i+1, IndexType(1),
                                         x+iX+incX, incX);
                    x[iX] *= conjugate(A[ldA*i]);
                }
            } else { /* diag==Unit */
                for (IndexType i=n-1, iX=i*incX; i>=0; --i, iX-=incX) {
                    IndexType len = min(k+1, n-i);

                    acxpy_generic(len-1, x[iX],
                                         A+ldA*i+1, IndexType(1),
                                         x+iX+incX, incX);
                }
            }
        } else { /* upLo==Lower */
            if (diag==NonUnit) {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType iY = max(i-k, IndexType(0))*incX;

                    acxpy_generic(len-1, x[iX],
                                         A+ldA*i+iA, IndexType(1),
                                         x+iY, incX);
                    x[iX] *= conjugate(A[ldA*i+iA+len-1]);
                }
            } else {
                for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                    IndexType iA = max(k-i, IndexType(0));
                    IndexType len = min(k, i) + 1;
                    IndexType iY = max(i-k, IndexType(0))*incX;

                    acxpy_generic(len-1, x[iX],
                                         A+ldA*i+iA, IndexType(1),
                                         x+iY, incX);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename MA, typename VX>
void
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const MA *A, IndexType ldA,
     VX *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("tbmv_generic");

    if (n==0) {
        return;
    }

    tbmv_generic(order, upLo, transA, diag, n, k, A, ldA, x, incX);
}


#ifdef HAVE_CBLAS

// stbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const float *A, IndexType ldA,
     float *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_stbmv");

    cblas_stbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n, k,
                A, ldA,
                x, incX);
}

// dtbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const double *A, IndexType ldA,
     double *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dtbmv");

    cblas_dtbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n, k,
                A, ldA,
                x, incX);
}

// ctbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const ComplexFloat *A, IndexType ldA,
     ComplexFloat *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ctbmv");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("tbmv_generic");
        tbmv_generic(order, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }

    cblas_ctbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n, k,
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<float *>(x), incX);
}

// ztbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const ComplexDouble *A, IndexType ldA,
     ComplexDouble *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ztbmv");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("tbmv_generic");
        tbmv_generic(order, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }

    cblas_ztbmv(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                n, k,
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<double *>(x), incX);
}
#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_TBMV_TCC

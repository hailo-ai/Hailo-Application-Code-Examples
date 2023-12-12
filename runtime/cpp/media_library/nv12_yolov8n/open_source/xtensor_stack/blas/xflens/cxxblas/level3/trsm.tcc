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

#ifndef CXXBLAS_LEVEL3_TRSM_TCC
#define CXXBLAS_LEVEL3_TRSM_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB>
void
trsm_generic(StorageOrder order, Side sideA, StorageUpLo upLoA,
             Transpose transA, Diag diagA,
             IndexType m, IndexType n,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             MB *B, IndexType ldB)
{
    if (order==ColMajor) {
        sideA = (sideA==Left) ? Right : Left;
        upLoA = (upLoA==Upper) ? Lower : Upper;
        trsm_generic(RowMajor, sideA, upLoA, transA, diagA, n, m,
                     alpha, A, ldA, B, ldB);
        return;
    }
    if (sideA==Right) {
        transA = Transpose(transA^Trans);
        for (IndexType i=0; i<m; ++i) {
            trsv(order, upLoA, transA, diagA, n, A, ldA, B+i*ldB, IndexType(1));
        }
    }
    if (sideA==Left) {
        for (IndexType j=0; j<n; ++j) {
            trsv(order, upLoA, transA, diagA, m, A, ldA, B+j, ldB);
        }
    }
    gescal(order, m, n, alpha, B, ldB);
}

template <typename IndexType, typename ALPHA, typename MA, typename MB>
void
trsm(StorageOrder order, Side side, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType m, IndexType n,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     MB *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("trsm_generic");

    trsm_generic(order, side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
}

#ifdef HAVE_CBLAS

// strsm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsm(StorageOrder order, Side side, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType m, IndexType n,
     float alpha,
     const float *A, IndexType ldA,
     float *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_strsm");

    cblas_strsm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                m, n,
                alpha,
                A, ldA,
                B, ldB);
}

// dtrsm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsm(StorageOrder order, Side side, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType m, IndexType n,
     double alpha,
     const double *A, IndexType ldA,
     double *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dtrsm");

    cblas_dtrsm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                m, n,
                alpha,
                A, ldA,
                B, ldB);
}

// ctrsm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsm(StorageOrder order, Side side, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType m, IndexType n,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     ComplexFloat *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ctrsm");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("trsm_generic");
        trsm_generic(order, side, upLo, transA, diag,
                     m, n, alpha, A, ldA, B, ldB);
        return;
    }

    cblas_ctrsm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                m, n,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<float *>(B), ldB);
}

// ztrsm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsm(StorageOrder order, Side side, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType m, IndexType n,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     ComplexDouble *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ztrsm");

    if (transA==Conj) {
        CXXBLAS_DEBUG_OUT("trsm_generic");
        trsm_generic(order, side, upLo, transA, diag,
                     m, n, alpha, A, ldA, B, ldB);
        return;
    }

    cblas_ztrsm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(diag),
                m, n,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<double *>(B), ldB);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL3_TRSM_TCC


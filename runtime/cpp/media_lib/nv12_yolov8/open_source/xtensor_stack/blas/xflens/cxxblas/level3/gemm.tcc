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

#ifndef CXXBLAS_LEVEL3_GEMM_TCC
#define CXXBLAS_LEVEL3_GEMM_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
void
gemm_generic(StorageOrder order,
             Transpose transA, Transpose transB,
             IndexType m, IndexType n, IndexType k,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const MB *B, IndexType ldB,
             const BETA &beta,
             MC *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("gemm_generic");

    if ((m==0) || (n==0)) {
        return;
    }
    if (order==ColMajor) {
        gemm_generic(RowMajor, transB, transA,
                     n, m, k, alpha,
                     B, ldB, A, ldA,
                     beta,
                     C, ldC);
        return;
    }

    gescal_init(order, m, n, beta, C, ldC);
    if (alpha==ALPHA(0)) {
        return;
    }
    if ((transA==NoTrans) && (transB==NoTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, NoTrans, m, k, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==NoTrans) && (transB==Conj)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, NoTrans, Conj, m, k, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==NoTrans) && (transB==Trans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, NoTrans, m, k, alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==NoTrans) && (transB==ConjTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, NoTrans, Conj, m, k,
                 alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }

    if ((transA==Conj) && (transB==NoTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, NoTrans, m, k, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==Conj) && (transB==Conj)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Conj, Conj, m, k, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==Conj) && (transB==Trans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Conj, m, k, alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==Conj) && (transB==ConjTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Conj, Conj, m, k, alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }

    if ((transA==Trans) && (transB==NoTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Trans, k, m, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==Trans) && (transB==Conj)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Trans, Conj, k, m, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==Trans) && (transB==Trans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Trans, k, m, alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==Trans) && (transB==ConjTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, Trans, Conj, k, m,
                 alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }

    if ((transA==ConjTrans) && (transB==NoTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, ConjTrans, k, m, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==ConjTrans) && (transB==Conj)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, ConjTrans, k, m, alpha, A, ldA, B+l, ldB,
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==ConjTrans) && (transB==Trans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, ConjTrans, k, m, alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }
    if ((transA==ConjTrans) && (transB==ConjTrans)) {
        for (IndexType l=0; l<n; ++l) {
            gemv(order, ConjTrans, Conj, k, m,
                 alpha, A, ldA, B+l*ldB, IndexType(1),
                 BETA(1), C+l, ldC);
        }
    }
}

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
void
gemm(StorageOrder order,
     Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const MB *B, IndexType ldB,
     const BETA &beta,
     MC *C, IndexType ldC)
{
    gemm_generic(order, transA, transB, m, n, k,
                 alpha, A, ldA, B, ldB,
                 beta,
                 C, ldC);
}

#ifdef HAVE_CBLAS

// sgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order,
     Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     float alpha,
     const float *A, IndexType ldA,
     const float *B, IndexType ldB,
     float beta,
     float *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sgemm");

    cblas_sgemm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(transB),
                m, n, k,
                alpha,
                A, ldA,
                B, ldB,
                beta,
                C, ldC);
}

// dgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order,
     Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     double alpha,
     const double *A, IndexType ldA,
     const double *B, IndexType ldB,
     double beta,
     double *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dgemm");

    cblas_dgemm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(transB),
                m, n, k,
                alpha,
                A, ldA,
                B, ldB,
                beta,
                C, ldC);
}

// cgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order,
     Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat *B, IndexType ldB,
     const ComplexFloat &beta,
     ComplexFloat *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cgemm");

    if (transA==Conj || transB==Conj) {
        CXXBLAS_DEBUG_OUT("gemm_generic");
        gemm_generic(order, transA, transB, m, n, k,
                    alpha, A, ldA, B, ldB,
                    beta,
                    C, ldC);
        return;
    }

    cblas_cgemm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(transB),
                m, n, k,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<const float *>(B), ldB,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(C), ldC);
}

// zgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order,
     Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble *B, IndexType ldB,
     const ComplexDouble &beta,
     ComplexDouble *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zgemm");

    if (transA==Conj || transB==Conj) {
        CXXBLAS_DEBUG_OUT("gemm_generic");
        gemm_generic(order, transA, transB, m, n, k,
                    alpha, A, ldA, B, ldB,
                    beta,
                    C, ldC);
        return;
    }

    cblas_zgemm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(transA), CBLAS::getCblasType(transB),
                m, n, k,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<const double *>(B), ldB,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(C), ldC);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL3_GEMM_TCC
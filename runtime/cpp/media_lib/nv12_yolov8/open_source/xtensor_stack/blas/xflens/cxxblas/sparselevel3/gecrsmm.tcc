/*
 *   Copyright (c) 2013, Klaus Pototzky
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

#ifndef CXXBLAS_SPARSELEVEL3_GECRSMM_TCC
#define CXXBLAS_SPARSELEVEL3_GECRSMM_TCC 1

#include "xflens/cxxblas/auxiliary/auxiliary.h"
#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_GECRSMM 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
void
gecrsmm(Transpose        transA,
        IndexType        m,
        IndexType        n,
        IndexType        k,
        const ALPHA      &alpha,
        const MA         *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const MB         *B,
        IndexType        ldB,
        const BETA       &beta,
        MC               *C,
        IndexType        ldC)
{
    CXXBLAS_DEBUG_OUT("gecrsmm_generic");

    for (IndexType i=0; i<n; ++i) {
         gecrsmv(transA, m, k, alpha, A, ia, ja, B+i*ldB, beta, C+i*ldC);
    }

    return;
}

#ifdef HAVE_SPARSEBLAS

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmm(Transpose        transA,
        IndexType        m,
        IndexType        n,
        IndexType        k,
        const float      &alpha,
        const float      *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const float      *B,
        IndexType        ldB,
        const float      &beta,
        float            *C,
        IndexType        ldC)
{

    CXXBLAS_DEBUG_OUT("gecrsmm -> [" BLAS_IMPL "] scsrmm");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmm<IndexType, float, float,
                            float, float,
                            float>
                            (transA, m, n, k, alpha,
                             A, ia, ja, B, ldB, beta, C, ldC);
         return;

    }
    char transA_ = getF77BlasChar(transA);

    mkl_scsrmm(&transA_,
               &m, &n, &k,
               &alpha, &matdescra[0],
               A, ja, ia, ia+1,
               B, ldB,
               &beta, C, ldC);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmm(Transpose        transA,
        IndexType        m,
        IndexType        n,
        IndexType        k,
        const double     &alpha,
        const double     *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const double     *B,
        IndexType        ldB,
        const double     &beta,
        double           *C,
        IndexType        ldC)
{

    CXXBLAS_DEBUG_OUT("gecrsmm -> [" BLAS_IMPL "] dcsrmm");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmm<IndexType, double, double,
                            double, double,
                            double>
                            (transA, m, n, k, alpha,
                             A, ia, ja, B, ldC, beta, C, ldC);
         return;

    }
    char transA_ = getF77BlasChar(transA);

    mkl_dcsrmm(&transA,
               &m, &n, &k,
               &alpha, &matdescra[0],
               A, ja, ia, ia+1,
               B, ldB,
               &beta, C, ldC);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmm(Transpose               transA,
        IndexType               m,
        IndexType               n,
        IndexType               k,
        const ComplexFloat      &alpha,
        const ComplexFloat      *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexFloat      *B,
        IndexType               ldB,
        const ComplexFloat      &beta,
        ComplexFloat            *C,
        IndexType               ldC)
{

    CXXBLAS_DEBUG_OUT("gecrsmm -> [" BLAS_IMPL "] ccsrmm");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmm<IndexType, ComplexFloat, ComplexFloat,
                            ComplexFloat, ComplexFloat,
                            ComplexFloat>
                            (transA, m, n, k, alpha,
                             A, ia, ja, B, ldB, beta, C, ldC);
         return;

    }
    char transA_ = getF77BlasChar(transA);

    if (transA==Conj) {
      transA_ = 'C';
      mkl_ccscmm(&transA_,
                 &m, &n, k,
                 reinterpret_cast<const float*>(&alpha), &matdescra[0],
                 reinterpret_cast<const float*>(A), ja, ia, ia+1,
                 reinterpret_cast<const float*>(B), ldB,
                 reinterpret_cast<const float*>(&beta),
                 reinterpret_cast<float*>(C), ldC);
    } else {
      mkl_ccsrmm(&transA_,
                 &m, &n, &k,
                 reinterpret_cast<const float*>(&alpha), &matdescra[0],
                 reinterpret_cast<const float*>(A), ja, ia, ia+1,
                 reinterpret_cast<const float*>(B), ldB,
                 reinterpret_cast<const float*>(&beta),
                 reinterpret_cast<float*>(C), ldC);
    }
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmm(Transpose               transA,
        IndexType               m,
        IndexType               n,
        IndexType               k,
        const ComplexDouble     &alpha,
        const ComplexDouble     *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexDouble     *B,
        IndexType               ldB,
        const ComplexDouble     &beta,
        ComplexDouble           *C,
        IndexType               ldC)
{

    CXXBLAS_DEBUG_OUT("gecrsmm -> [" BLAS_IMPL "] zcsrmm");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmm<IndexType, ComplexDouble, ComplexDouble,
                            ComplexDouble, ComplexDouble,
                            ComplexDouble>
                            (transA, m, n, k, alpha,
                             A, ia, ja, B, ldB, beta, C, ldC);
         return;

    }

    char transA_ = getF77BlasChar(transA);

    if (transA==Conj) {
      transA_ = 'C';
      mkl_zcscmm(&transA_,
                 &m, &n, &k,
                 reinterpret_cast<const double*>(&alpha), &matdescra[0],
                 reinterpret_cast<const double*>(A), ja, ia, ia+1,
                 reinterpret_cast<const double*>(B), ldB,
                 reinterpret_cast<const double*>(&beta),
                 reinterpret_cast<double*>(C), ldC);
    } else {
      mkl_zcsrmm(&transA_,
                 &m, &n, &k,
                 reinterpret_cast<const double*>(&alpha), &matdescra[0],
                 reinterpret_cast<const double*>(A), ja, ia, ia+1,
                 reinterpret_cast<const double*>(B), ldB,
                 reinterpret_cast<const double*>(&beta),
                 reinterpret_cast<double*>(C), ldC);
    }
}

#endif

} // namespace cxxblas

#endif // CXXBLAS_SPARSELEVEL3_GECRSMM_TCC

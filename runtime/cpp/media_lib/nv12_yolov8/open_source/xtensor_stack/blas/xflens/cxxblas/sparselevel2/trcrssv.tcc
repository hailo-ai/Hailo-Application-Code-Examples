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

#ifndef CXXBLAS_SPARSELEVEL2_TRCRSSV_TCC
#define CXXBLAS_SPARSELEVEL2_TRCRSSV_TCC 1

#include "xflens/cxxblas/typedefs.h"

namespace cxxblas {

#ifdef HAVE_SPARSEBLAS

#define HAVE_CXXBLAS_TRCRSSV 1

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trcrssv(StorageUpLo      upLo,
        Transpose        trans,
        IndexType        n,
        const float      &alpha,
        const float      *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const float      *x,
        float            *y)
{
    CXXBLAS_DEBUG_OUT("trcrssv -> [" BLAS_IMPL "] scsrsv");

    char matdescra[5] = { "T*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    ASSERT(matdescra[3]!='E') ;

    char transA = getF77BlasChar(trans);

    mkl_scsrsv(&transA,
               &n,
               &alpha, &matdescra[0],
               A, ja, ia, ia+1,
               x, y);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trcrssv(StorageUpLo      upLo,
        Transpose        trans,
        IndexType        n,
        const double     &alpha,
        const double     *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const double     *x,
        double           *y)
{
    CXXBLAS_DEBUG_OUT("trcrssv -> [" BLAS_IMPL "] dcsrsv");

    char matdescra[5] = { "T*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    ASSERT(matdescra[3]!='E') ;

    char transA = getF77BlasChar(trans);

    mkl_dcsrsv(&transA,
               &n,
               &alpha, &matdescra[0],
               A, ja, ia, ia+1,
               x, y);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trcrssv(StorageUpLo             upLo,
        Transpose               trans,
        IndexType               n,
        const ComplexFloat      &alpha,
        const ComplexFloat      *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexFloat      *x,
        ComplexFloat            *y)
{
    CXXBLAS_DEBUG_OUT("trcrssv -> [" BLAS_IMPL "] ccsrsv");

    char matdescra[5] = { "T*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    ASSERT(matdescra[3]!='E') ;

    char transA = getF77BlasChar(trans);

    mkl_ccsrsv(&transA,
               &n,
               reinterpret_cast<const float*>(&alpha), &matdescra[0],
               reinterpret_cast<const float*>(A), ja, ia, ia+1,
               reinterpret_cast<const float*>(x),
               reinterpret_cast<float*>(y));

}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trcrssv(StorageUpLo             upLo,
        Transpose               trans,
        IndexType               n,
        const ComplexDouble     &alpha,
        const ComplexDouble     *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexDouble     *x,
        ComplexDouble           *y)
{
    CXXBLAS_DEBUG_OUT("trcrssv -> [" BLAS_IMPL "] zcsrsv");

    char matdescra[5] = { "T*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    ASSERT(matdescra[3]!='E');

    char transA = getF77BlasChar(trans);

    mkl_zcsrsv(&transA,
               &n,
               reinterpret_cast<const double*>(&alpha), &matdescra[0],
               reinterpret_cast<const double*>(A), ja, ia, ia+1,
               reinterpret_cast<const double*>(x),
               reinterpret_cast<double*>(y));

}

#endif


} // namespace cxxblas

#endif // CXXBLAS_SPARSELEVEL2_TRCRSSV_TCC

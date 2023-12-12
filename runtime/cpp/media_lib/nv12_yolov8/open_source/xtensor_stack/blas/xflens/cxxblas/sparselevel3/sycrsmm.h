/*
 *   Copyright (c) 2007-2012, Michael Lehn
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

#ifndef CXXBLAS_SPARSELEVEL3_SYCRSMM_H
#define CXXBLAS_SPARSELEVEL3_SYCRSMM_H 1

#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_SYCRSMM 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
    void
    sycrsmm(StorageUpLo      upLo,
            IndexType        m,
            IndexType        n,
            const ALPHA      &alpha,
            const MA         *A,
            const IndexType  *ia,
            const IndexType  *ja,
            const MB         *B,
            IndexType        ldB,
            const BETA       &beta,
            MC               *C,
            IndexType        ldC);


#ifdef HAVE_SPARSEBLAS

template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    sycrsmm(StorageUpLo      upLo,
            IndexType        m,
            IndexType        n,
            const float      &alpha,
            const float      *A,
            const IndexType  *ia,
            const IndexType  *ja,
            const float      *B,
            IndexType        ldB,
            const float      &beta,
            float            *C,
            IndexType        ldC);

template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    sycrsmm(StorageUpLo      upLo,
            IndexType        m,
            IndexType        n,
            const double     &alpha,
            const double     *A,
            const IndexType  *ia,
            const IndexType  *ja,
            const double     *B,
            IndexType        ldB,
            const double     &beta,
            double           *C,
            IndexType        ldC);


template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    sycrsmm(StorageUpLo             upLo,
            IndexType               m,
            IndexType               n,
            const ComplexFloat      &alpha,
            const ComplexFloat      *A,
            const IndexType         *ia,
            const IndexType         *ja,
            const ComplexFloat      *B,
            IndexType               ldB,
            const ComplexFloat      &beta,
            ComplexFloat            *C,
            IndexType               ldC);

template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    sycrsmm(StorageUpLo             upLo,
            IndexType               m,
            IndexType               n,
            const ComplexDouble     &alpha,
            const ComplexDouble     *A,
            const IndexType         *ia,
            const IndexType         *ja,
            const ComplexDouble     *B,
            IndexType               ldB,
            const ComplexDouble     &beta,
            ComplexDouble           *C,
            IndexType               ldC);

#endif // HAVE_SPARSEBLAS

} // namespace cxxblas

#endif // CXXBLAS_SPARSELEVEL3_SYCRSMM_H

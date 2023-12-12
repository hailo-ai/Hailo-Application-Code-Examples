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

#ifndef CXXBLAS_SPARSELEVEL2_SYCCSMV_TCC
#define CXXBLAS_SPARSELEVEL2_SYCCSMV_TCC 1

#include "xflens/cxxblas/auxiliary/auxiliary.h"
#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_SYCCSMV 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
syccsmv(StorageUpLo      upLo,
        IndexType        n,
        const ALPHA      &alpha,
        const MA         *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const VX         *x,
        const BETA       &beta,
        VY               *y)
{
    CXXBLAS_DEBUG_OUT("syccsmv_generic");

//
//  The correct index base of the CCS matrix is stored in first Element of ja
//
    --ja;
    ia -= ja[1];
    A  -= ja[1];

//
//  Let x, y be one-based; x_, y_ get correct index base ja[1]
//
    const VX *x_ = x - ja[1];
    VY       *y_ = y - ja[1];
    --x;
    --y;

    if (beta==BETA(0)) {
        for (int i=1; i<=n; ++i) {
            y[i] = 0;
        }
    } else if (beta!=BETA(1)) {
        for (int i=1; i<=n; ++i) {
            y[i] *= beta;
        }
    }

    if (upLo==Lower) {
        for (int j=1, J=ja[1]; j<=n; ++j, ++J) {
            if (ja[j]<ja[j+1]) {
                int k=ja[j];
                if (ia[k]==J) {
                    y[j]      += alpha*A[k]*x_[ia[k]];
                } else {
                    y[j]      += alpha*A[k]*x_[ia[k]];
                    y_[ia[k]] += alpha*A[k]*x[j];
                }
                for (k=ja[j]+1; k<ja[j+1]; ++k) {
                    y[j]      += alpha*A[k]*x_[ia[k]];
                    y_[ia[k]] += alpha*A[k]*x[j];
                }
            }
        }
    } else {
        for (int j=1, J=ja[1]; j<=n; ++j, ++J) {
            if (ja[j]<ja[j+1]) {
                int k;
                for (k=ja[j]; k<ja[j+1]-1; ++k) {
                    y[j]      += alpha*A[k]*x_[ia[k]];
                    y_[ia[k]] += alpha*A[k]*x[j];
                }
                if (ia[k]==J) {
                    y[j]      += alpha*A[k]*x_[ia[k]];
                } else {
                    y[j]      += alpha*A[k]*x_[ia[k]];
                    y_[ia[k]] += alpha*A[k]*x[j];
                }
            }
        }
    }
}

#ifdef HAVE_SPARSEBLAS

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syccsmv(StorageUpLo      upLo,
        IndexType        n,
        const float      &alpha,
        const float      *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const float      *x,
        const float      &beta,
        float            *y)
{
    CXXBLAS_DEBUG_OUT("syccsmv -> [" BLAS_IMPL "] scscmv");

    char matdescra[5] = { "S*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         syccsmv<IndexType, float, float,
                            float, float,
                            float>
                            (upLo, n, alpha, A, ia, ja, x, beta, y);
         return;
    }

    char transA = 'N';

    mkl_scscmv(&transA,
               &n, &n,
               &alpha, &matdescra[0],
               A, ia, ja, ja+1,
               x,
               &beta, y);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syccsmv(StorageUpLo      upLo,
        IndexType        n,
        const double     &alpha,
        const double     *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const double     *x,
        const double     &beta,
        double           *y)
{
    CXXBLAS_DEBUG_OUT("syccsmv -> [" BLAS_IMPL "] dcscmv");

    char matdescra[5] = { "S*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         syccsmv<IndexType, double, double,
                            double, double,
                            double>
                            (upLo, n, alpha, A, ia, ja, x, beta, y);
         return;
    }

    char transA = 'N';

    mkl_dcscmv(&transA,
               &n, &n,
               &alpha, &matdescra[0],
               A, ia, ja, ja+1,
               x,
               &beta, y);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syccsmv(StorageUpLo             upLo,
        IndexType               n,
        const ComplexFloat      &alpha,
        const ComplexFloat      *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexFloat      *x,
        const ComplexFloat      &beta,
        ComplexFloat            *y)
{
    CXXBLAS_DEBUG_OUT("syccsmv -> [" BLAS_IMPL "] ccscmv");

    char matdescra[5] = { "S*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         syccsmv<IndexType, ComplexFloat, ComplexFloat,
                            ComplexFloat, ComplexFloat,
                            ComplexFloat>
                            (upLo, n, alpha, A, ia, ja, x, beta, y);
         return;
    }

    char transA = 'N';

    mkl_ccscmv(&transA,
               &n, &n,
               reinterpret_cast<const float*>(&alpha), &matdescra[0],
               reinterpret_cast<const float*>(A), ia, ja, ja+1,
               reinterpret_cast<const float*>(x),
               reinterpret_cast<const float*>(&beta),
               reinterpret_cast<float*>(y));

}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syccsmv(StorageUpLo             upLo,
        IndexType               n,
        const ComplexDouble     &alpha,
        const ComplexDouble     *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexDouble     *x,
        const ComplexDouble     &beta,
        ComplexDouble           *y)
{
    CXXBLAS_DEBUG_OUT("syccsmv -> [" BLAS_IMPL "] zcscmv");

    char matdescra[5] = { "S*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         syccsmv<IndexType, ComplexDouble, ComplexDouble,
                            ComplexDouble, ComplexDouble,
                            ComplexDouble>
                            (upLo, n, alpha, A, ia, ja, x, beta, y);
         return;
    }

    char transA = 'N';

    mkl_zcscmv(&transA,
               &n, &n,
               reinterpret_cast<const double*>(&alpha), &matdescra[0],
               reinterpret_cast<const double*>(A), ia, ja, ja+1,
               reinterpret_cast<const double*>(x),
               reinterpret_cast<const double*>(&beta),
               reinterpret_cast<double*>(y));

}

#endif

} // namespace cxxblas

#endif // CXXBLAS_SPARSELEVEL2_SYCCSMV_TCC

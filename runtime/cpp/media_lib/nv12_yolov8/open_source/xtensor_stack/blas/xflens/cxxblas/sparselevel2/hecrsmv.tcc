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

#ifndef CXXBLAS_SPARSELEVEL2_HECRSMV_TCC
#define CXXBLAS_SPARSELEVEL2_HECRSMV_TCC 1

#include "xflens/cxxblas/auxiliary/auxiliary.h"
#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_HECRSMV 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
hecrsmv(StorageUpLo      upLo,
        IndexType        n,
        const ALPHA      &alpha,
        const MA         *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const VX         *x,
        const BETA       &beta,
        VY               *y)
{
    CXXBLAS_DEBUG_OUT("hecrsmv_generic");

    using cxxblas::conjugate;

//
//  The correct index base of the CRS matrix is stored in first Element of ia
//
    --ia;
    ja -= ia[1];
    A  -= ia[1];

//
//  Let x, y be one-based; x_, y_ get correct index base ia[1]
//
    const VX *x_ = x - ia[1];
    VY       *y_ = y - ia[1];
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

    if (upLo==Upper) {
        for (int i=1, I=ia[1]; i<=n; ++i, ++I) {
            if (ia[i]<ia[i+1]) {
                int k=ia[i];
                if (ja[k]==I) {
                    y[i]      += alpha*cxxblas::real(A[k])*x_[ja[k]];
                } else {
                    y[i]      += alpha*A[k]*x_[ja[k]];
                    y_[ja[k]] += alpha*conjugate(A[k])*x[i];
                }
                for (k=ia[i]+1; k<ia[i+1]; ++k) {
                    y[i]      += alpha*A[k]*x_[ja[k]];
                    y_[ja[k]] += alpha*conjugate(A[k])*x[i];
                }
            }
        }
    } else {
        for (int i=1, I=ia[1]; i<=n; ++i, ++I) {
            if (ia[i]<ia[i+1]) {
                int k;
                for (k=ia[i]; k<ia[i+1]-1; ++k) {
                    y[i]      += alpha*A[k]*x_[ja[k]];
                    y_[ja[k]] += alpha*conjugate(A[k])*x[i];
                }
                if (ja[k]==I) {
                    y[i]      += alpha*cxxblas::real(A[k])*x_[ja[k]];
                } else {
                    y[i]      += alpha*A[k]*x_[ja[k]];
                    y_[ja[k]] += alpha*conjugate(A[k])*x[i];
                }
            }
        }
    }
}

#ifdef HAVE_SPARSEBLAS

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
hecrsmv(StorageUpLo             upLo,
        IndexType               n,
        const ComplexFloat      &alpha,
        const ComplexFloat      *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexFloat      *x,
        const ComplexFloat      &beta,
        ComplexFloat            *y)
{
    CXXBLAS_DEBUG_OUT("hecrsmv -> [" BLAS_IMPL "] ccsrmv");

    char matdescra[5] = { "H*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         hecrsmv<IndexType, ComplexFloat, ComplexFloat,
                            ComplexFloat, ComplexFloat,
                            ComplexFloat>
                            (upLo, n, alpha, A, ia, ja, x, beta, y);
         return;
    }

    char transA = 'N';

    mkl_ccsrmv(&transA,
               &n, &n,
               reinterpret_cast<const float*>(&alpha), &matdescra[0],
               reinterpret_cast<const float*>(A), ja, ia, ia+1,
               reinterpret_cast<const float*>(x),
               reinterpret_cast<const float*>(&beta),
               reinterpret_cast<float*>(y));

}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
hecrsmv(StorageUpLo             upLo,
        IndexType               n,
        const ComplexDouble     &alpha,
        const ComplexDouble     *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexDouble     *x,
        const ComplexDouble     &beta,
        ComplexDouble           *y)
{
    CXXBLAS_DEBUG_OUT("hecrsmv -> [" BLAS_IMPL "] zcsrmv");

    char matdescra[5] = { "H*N*" };
    matdescra[1] = getF77BlasChar(upLo);
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         hecrsmv<IndexType, ComplexDouble, ComplexDouble,
                            ComplexDouble, ComplexDouble,
                            ComplexDouble>
                            (upLo, n, alpha, A, ia, ja, x, beta, y);
         return;
    }

    char transA = 'N';

    mkl_zcsrmv(&transA,
              &n, &n,
              reinterpret_cast<const double*>(&alpha), &matdescra[0],
              reinterpret_cast<const double*>(A), ja, ia, ia+1,
              reinterpret_cast<const double*>(x),
              reinterpret_cast<const double*>(&beta),
              reinterpret_cast<double*>(y));

}

#endif

} // namespace cxxblas

#endif // CXXBLAS_SPARSELEVEL2_HECRSMV_TCC

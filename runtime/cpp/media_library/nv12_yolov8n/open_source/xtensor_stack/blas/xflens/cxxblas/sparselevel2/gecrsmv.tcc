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

#ifndef CXXBLAS_SPARSELEVEL2_GECRSMV_TCC
#define CXXBLAS_SPARSELEVEL2_GECRSMV_TCC 1

#include "xflens/cxxblas/auxiliary/auxiliary.h"
#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_GECRSMV 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
gecrsmv(Transpose        trans,
        IndexType        m,
        IndexType        n,
        const ALPHA      &alpha,
        const MA         *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const VX         *x,
        const BETA       &beta,
        VY               *y)
{
    CXXBLAS_DEBUG_OUT("gecrsmv_generic");

    using cxxblas::conjugate;

    const bool init  = (beta==BETA(0));
    const bool scale = (beta!=BETA(0) && beta!=BETA(1));

//
//  Index base of the CRS matrix is stored in first Element of ia
//
    --ia;
    ja -= ia[1];
    A  -= ia[1];

    if (trans==NoTrans) {
//
//      Make y one-based; set correct index base for x
//
        --y;
        x  -= ia[1];

        if (init) {
            for (int i=1; i<=m; ++i) {
                y[i] = VY(0);
                for (int k=ia[i]; k<ia[i+1]; ++k) {
                    y[i] += alpha*A[k]*x[ja[k]];
                }
            }
        } else if (scale) {
            for (int i=1; i<=m; ++i) {
                y[i] *= beta;
                for (int k=ia[i]; k<ia[i+1]; ++k) {
                    y[i] += alpha*A[k]*x[ja[k]];
                }
            }
        } else {
            for (int i=1; i<=m; ++i) {
                for (int k=ia[i]; k<ia[i+1]; ++k) {
                    y[i] += alpha*A[k]*x[ja[k]];
                }
            }
        }
    } else if (trans==Conj) {
//
//      Make y one-based; set correct index base for x
//
        --y;
        x  -= ia[1];

        if (init) {
            for (int i=1; i<=m; ++i) {
                y[i] = VY(0);
                for (int k=ia[i]; k<ia[i+1]; ++k) {
                    y[i] += alpha*conjugate(A[k])*x[ja[k]];
                }
            }
        } else if (scale) {
            for (int i=1; i<=m; ++i) {
                y[i] *= beta;
                for (int k=ia[i]; k<ia[i+1]; ++k) {
                    y[i] += alpha*conjugate(A[k])*x[ja[k]];
                }
            }
        } else {
            for (int i=1; i<=m; ++i) {
                for (int k=ia[i]; k<ia[i+1]; ++k) {
                    y[i] += alpha*conjugate(A[k])*x[ja[k]];
                }
            }
        }
    } else if (trans==Trans) {
//
//      Make y one-based
//
        --y;
        if (init) {
            for (int i=1; i<=n; ++i) {
                y[i] = VY(0);
            }
        } else if (scale) {
            for (int i=1; i<=n; ++i) {
                y[i] *=beta;
            }
        }
//
//      Set corret index base for y; make x one-based
//
        y += 1-ia[1];
        --x;

        for (int i=1; i<=m; ++i) {
            for (int k=ia[i]; k<ia[i+1]; ++k) {
                y[ja[k]] += alpha*A[k]*x[i];
            }
        }
    } else if (trans==ConjTrans) {
//
//      Make y one-based
//
        --y;
        if (init) {
            for (int i=1; i<=n; ++i) {
                y[i] = VY(0);
            }
        } else if (scale) {
            for (int i=1; i<=n; ++i) {
                y[i] *=beta;
            }
        }
//
//      Set corret index base for y; make x one-based
//
        y += 1-ia[1];
        --x;

        for (int i=1; i<=m; ++i) {
            for (int k=ia[i]; k<ia[i+1]; ++k) {
                y[ja[k]] += alpha*conjugate(A[k])*x[i];
            }
        }
    }
}

#ifdef HAVE_SPARSEBLAS

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmv(Transpose        trans,
        IndexType        m,
        IndexType        n,
        const float      &alpha,
        const float      *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const float      *x,
        const float      &beta,
        float            *y)
{
    CXXBLAS_DEBUG_OUT("gecrsmv -> [" BLAS_IMPL "] scsrmv");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmv<IndexType, float, float,
                            float, float,
                            float>
                            (trans, m, n, alpha, A, ia, ja, x, beta, y);
         return;

    }
    char transA = getF77BlasChar(trans);

    mkl_scsrmv(&transA,
               &m, &n,
               &alpha, &matdescra[0],
               A, ja, ia, ia+1,
               x,
               &beta, y);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmv(Transpose        trans,
        IndexType        m,
        IndexType        n,
        const double     &alpha,
        const double     *A,
        const IndexType  *ia,
        const IndexType  *ja,
        const double     *x,
        const double     &beta,
        double           *y)
{
    CXXBLAS_DEBUG_OUT("gecrsmv -> [" BLAS_IMPL "] dcsrmv");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmv<IndexType, double, double,
                            double, double,
                            double>
                            (trans, m, n, alpha, A, ia, ja, x, beta, y);
         return;

    }
    char transA = getF77BlasChar(trans);

    mkl_dcsrmv(&transA,
               &m, &n,
               &alpha, &matdescra[0],
               A, ja, ia, ia+1,
               x,
               &beta, y);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmv(Transpose               trans,
        IndexType               m,
        IndexType               n,
        const ComplexFloat      &alpha,
        const ComplexFloat      *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexFloat      *x,
        const ComplexFloat      &beta,
        ComplexFloat            *y)
{
    CXXBLAS_DEBUG_OUT("gecrsmv -> [" BLAS_IMPL "] ccsrmv");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmv<IndexType, ComplexFloat, ComplexFloat,
                            ComplexFloat, ComplexFloat,
                            ComplexFloat>
                            (trans, m, n, alpha, A, ia, ja, x, beta, y);
         return;

    }
    char transA = getF77BlasChar(trans);

    if (trans==Conj) {
      transA = 'C';
      mkl_ccscmv(&transA,
                &m, &n,
                reinterpret_cast<const float*>(&alpha), &matdescra[0],
                reinterpret_cast<const float*>(A), ja, ia, ia+1,
                reinterpret_cast<const float*>(x),
                reinterpret_cast<const float*>(&beta),
                reinterpret_cast<float*>(y));
    } else {
      mkl_ccsrmv(&transA,
                &m, &n,
                reinterpret_cast<const float*>(&alpha), &matdescra[0],
                reinterpret_cast<const float*>(A), ja, ia, ia+1,
                reinterpret_cast<const float*>(x),
                reinterpret_cast<const float*>(&beta),
                reinterpret_cast<float*>(y));
    }
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gecrsmv(Transpose               trans,
        IndexType               m,
        IndexType               n,
        const ComplexDouble     &alpha,
        const ComplexDouble     *A,
        const IndexType         *ia,
        const IndexType         *ja,
        const ComplexDouble     *x,
        const ComplexDouble     &beta,
        ComplexDouble           *y)
{
    CXXBLAS_DEBUG_OUT("gecrsmv -> [" BLAS_IMPL "] zcsrmv");

    char matdescra[5] = { "G***" };
    matdescra[3] = getIndexBaseChar(ia[0]);

    if (matdescra[3]=='E') {
         gecrsmv<IndexType, ComplexDouble, ComplexDouble,
                            ComplexDouble, ComplexDouble,
                            ComplexDouble>
                            (trans, m, n, alpha, A, ia, ja, x, beta, y);
         return;

    }

    char transA = getF77BlasChar(trans);

    if (trans==Conj) {
      transA = 'C';
      mkl_zcscmv(&transA,
                &m, &n,
                reinterpret_cast<const double*>(&alpha), &matdescra[0],
                reinterpret_cast<const double*>(A), ja, ia, ia+1,
                reinterpret_cast<const double*>(x),
                reinterpret_cast<const double*>(&beta),
                reinterpret_cast<double*>(y));
    } else {
      mkl_zcsrmv(&transA,
                &m, &n,
                reinterpret_cast<const double*>(&alpha), &matdescra[0],
                reinterpret_cast<const double*>(A), ja, ia, ia+1,
                reinterpret_cast<const double*>(x),
                reinterpret_cast<const double*>(&beta),
                reinterpret_cast<double*>(y));
    }
}

#endif

} // namespace cxxblas

#endif // CXXBLAS_SPARSELEVEL2_GECRSMV_TCC

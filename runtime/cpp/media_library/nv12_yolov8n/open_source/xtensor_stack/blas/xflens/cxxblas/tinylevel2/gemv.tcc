/*
 *   Copyright (c) 2012, Michael Lehn
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

#ifndef CXXBLAS_TINYLEVEL2_GEMV_TCC
#define CXXBLAS_TINYLEVEL2_GEMV_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <int m, int n,
          typename MA, int ldA,
          typename VX, int incX,
          typename VY, int incY>
void
gemv_n(MA alpha, const MA *A, const VX *x, VY beta, VY *y)
{
    CXXBLAS_DEBUG_OUT("gemv_n [tiny]");

    for (int i=0, iY=0; i<m; ++i, iY+=incY) {
        y[iY] *= beta;
        for (int j=0, jX=0; j<n; ++j, jX+=incX) {
            y[iY] += alpha * A[i*ldA+j] * x[jX];
        }
    }
}

template <int m, int n,
          typename MA, int ldA,
          typename VX, int incX,
          typename VY, int incY>
void
gemv_c(MA alpha, const MA *A, const VX *x, VY beta, VY *y)
{
    CXXBLAS_DEBUG_OUT("gemv_c [tiny]");

    for (int i=0, iY=0; i<m; ++i, iY+=incY) {
        y[iY] *= beta;
        for (int j=0, jX=0; j<n; ++j, jX+=incX) {
            y[iY] += alpha * conjugate(A[i*ldA+j]) * x[jX];
        }
    }
}

template <int m, int n,
          typename MA, int ldA,
          typename VX, int incX,
          typename VY, int incY>
void
gemv_t(MA alpha, const MA *A, const VX *x, VY beta, VY *y)
{
    CXXBLAS_DEBUG_OUT("gemv_t [tiny]");

    for (int j=0, jY=0; j<n; ++j, jY+=incY) {
        y[jY] *= beta;
    }
    for (int i=0, iX=0; i<m; ++i, iX+=incY) {
        for (int j=0, jY=0; j<n; ++j, jY+=incY) {
            y[jY] += alpha * A[i*ldA+j] * x[iX];
        }
    }
}

template <int m, int n,
          typename MA, int ldA,
          typename VX, int incX,
          typename VY, int incY>
void
gemv_ct(MA alpha, const MA *A, const VX *x, VY beta, VY *y)
{
    CXXBLAS_DEBUG_OUT("gemv_t [tiny]");

    for (int j=0, jY=0; j<n; ++j, jY+=incY) {
        y[jY] *= beta;
    }
    for (int i=0, iX=0; i<m; ++i, iX+=incY) {
        for (int j=0, jY=0; j<n; ++j, jY+=incY) {
            y[jY] += alpha * conjugate(A[i*ldA+j]) * x[iX];
        }
    }
}

template <int m, int n,
          typename MA, int ldA,
          typename VX, int incX,
          typename VY, int incY>
void
gemv(Transpose trans, MA alpha, const MA *A, const VX *x, VY beta, VY *y)
{
    CXXBLAS_DEBUG_OUT("gemv [tiny]");

    if (trans==NoTrans) {
        gemv_n<m, n, MA, ldA, VX, incX, VY, incY>(alpha, A, x, beta, y);
    } else if (trans==Trans) {
        gemv_t<m, n, MA, ldA, VX, incX, VY, incY>(alpha, A, x, beta, y);
    } else if (trans==Conj){
        gemv_c<m, n, MA, ldA, VX, incX, VY, incY>(alpha, A, x, beta, y);
    } else if (trans==ConjTrans){
        gemv_ct<m, n, MA, ldA, VX, incX, VY, incY>(alpha, A, x, beta, y);
    }
}

} // namespace cxxblas

#endif // CXXBLAS_TINYLEVEL2_GEMV_TCC

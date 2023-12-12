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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_ACXPY_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_ACXPY_TCC 1

#include <cstdio>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename X, typename Y>
void
acxpy_generic(IndexType n, const ALPHA &alpha, const X *x,
              IndexType incX, Y *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("acxpy_generic");

    for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
        y[iY] += alpha*conjugate(x[iX]);
    }
}

template <typename IndexType, typename ALPHA, typename X, typename Y>
void
acxpy(IndexType n, const ALPHA &alpha, const X *x,
      IndexType incX, Y *y, IndexType incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    acxpy_generic(n, alpha, x, incX, y, incY);
}


#ifdef HAVE_CBLAS

template <typename IndexType>
void
acxpy(IndexType n, const float &alpha, const std::complex<float> *x,
      IndexType incX, std::complex<float> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("acxpy_generic (complex float)");

    cblas_saxpy(n, alpha,
                reinterpret_cast<const float *>(x), 2*incX,
                reinterpret_cast<float *>(y),
                2*incY);
    cblas_saxpy(n, -alpha,
                reinterpret_cast<const float *>(x)+1, 2*incX,
                reinterpret_cast<float *>(y)+1, 2*incY);

}

template <typename IndexType>
void
acxpy(IndexType n, const double &alpha, const std::complex<double> *x,
      IndexType incX, std::complex<double> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("acxpy_generic (complex double)");

    cblas_daxpy(n, alpha,
                reinterpret_cast<const double *>(x), 2*incX,
                reinterpret_cast<double *>(y), 2*incY);
    cblas_daxpy(n, -alpha,
                reinterpret_cast<const double *>(x)+1, 2*incX,
                reinterpret_cast<double *>(y)+1, 2*incY);

}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_ACXPY_TCC

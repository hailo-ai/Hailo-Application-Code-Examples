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

#ifndef CXXBLAS_LEVEL1_SCAL_TCC
#define CXXBLAS_LEVEL1_SCAL_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename Y>
void
scal_generic(IndexType n, const ALPHA &alpha, Y *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("scal_generic");

    for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
        y[iY] *= alpha;
    }
}

template <typename IndexType, typename ALPHA, typename Y>
void
scal_init_generic(IndexType n, const ALPHA &alpha, Y *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("scal_init_generic");

    if (alpha == ALPHA(0)) {
        for (IndexType i=0, iY=0; i<n; ++i, iY+=incY) {
            y[iY] = 0;
        }
    }
    else {
        scal_generic(n, alpha, y, incY);
    }
}

template <typename IndexType, typename ALPHA, typename Y>
void
scal(IndexType n, const ALPHA &alpha, Y *y, IndexType incY)
{
    if (incY<0) {
        y -= incY*(n-1);
    }
    scal_generic(n, alpha, y, incY);
}

template <typename IndexType, typename ALPHA, typename Y>
void
scal_init(IndexType n, const ALPHA &alpha, Y *y, IndexType incY)
{
    if (incY<0) {
        y -= incY*(n-1);
    }
    scal_init_generic(n, alpha, y, incY);
}

#ifdef HAVE_CBLAS

// sscal
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
scal(IndexType n, float alpha, float *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sscal");

    cblas_sscal(n, alpha, x, incX);
}

// dscal
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
scal(IndexType n, double alpha, double *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dscal");

    cblas_dscal(n, alpha, x, incX);
}

// cscal
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
scal(IndexType n, const ComplexFloat &alpha, ComplexFloat *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cscal");

    cblas_cscal(n, reinterpret_cast<const float *>(&alpha),
                   reinterpret_cast<float *>(x), incX);
}

// zscal
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
scal(IndexType n, const ComplexDouble &alpha, ComplexDouble *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zscal");

    cblas_zscal(n, reinterpret_cast<const double *>(&alpha),
                   reinterpret_cast<double *>(x), incX);
}

// csscal
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
scal(IndexType n, float alpha, ComplexFloat *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_csscal");

    cblas_csscal(n, alpha, reinterpret_cast<float *>(x), incX);
}

// zdscal
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
scal(IndexType n, double alpha, ComplexDouble *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdscal");

    cblas_zdscal(n, alpha, reinterpret_cast<double *>(x), incX);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_SCAL_TCC

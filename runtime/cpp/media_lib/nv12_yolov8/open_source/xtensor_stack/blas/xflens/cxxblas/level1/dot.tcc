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

#ifndef CXXBLAS_LEVEL1_DOT_TCC
#define CXXBLAS_LEVEL1_DOT_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename X, typename Y, typename Result>
void
dotu_generic(IndexType n,
             const X *x, IndexType incX, const Y *y, IndexType incY,
             Result &result)
{
    CXXBLAS_DEBUG_OUT("dotu_generic");

    result = Result(0);
    for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
        result += Result(x[iX])*Result(y[iY]);
    }
}


template <typename IndexType, typename X, typename Y, typename Result>
void
dot_generic(IndexType n,
            const X *x, IndexType incX, const Y *y, IndexType incY,
            Result &result)
{
    CXXBLAS_DEBUG_OUT("dot_generic");

    result = Result(0);
    for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
        result += Result(conjugate(x[iX]))*Result(y[iY]);
    }
}

//------------------------------------------------------------------------------

template <typename IndexType, typename X, typename Y, typename Result>
void
dotu(IndexType n,
     const X *x, IndexType incX, const Y *y, IndexType incY,
     Result &result)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    dotu_generic(n, x, incX, y, incY, result);
}

template <typename IndexType, typename X, typename Y, typename Result>
void
dot(IndexType n,
    const X *x, IndexType incX, const Y *y, IndexType incY,
    Result &result)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    dot_generic(n, x, incX, y, incY, result);
}

#ifdef HAVE_CBLAS

// sdsdot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
sdot(IndexType n, float alpha,
     const float *x, IndexType incX,
     const float *y, IndexType incY,
     float &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sdsdot");

    result = cblas_sdsdot(n, alpha, x, incX, y, incY);
}

// dsdot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const float *x, IndexType incX,
    const float *y, IndexType incY,
    double &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dsdot");

    result = cblas_dsdot(n, x, incX, y, incY);
}

// sdot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const float *x, IndexType incX,
    const float  *y, IndexType incY,
    float &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_sdot");

    result = cblas_sdot(n, x, incX, y, incY);
}

// ddot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const double *x, IndexType incX,
    const double *y, IndexType incY,
    double &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ddot");

    result = cblas_ddot(n, x, incX, y, incY);
}

// cdotu_sub
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const ComplexFloat  *x, IndexType incX,
     const ComplexFloat  *y, IndexType incY,
     ComplexFloat &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cdotu_sub");

    cblas_cdotu_sub(n, reinterpret_cast<const float *>(x), incX,
                       reinterpret_cast<const float *>(y), incY,
                       reinterpret_cast<float *>(&result));
}


// cdotc_sub
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const ComplexFloat  *x, IndexType incX,
    const ComplexFloat  *y, IndexType incY,
    ComplexFloat &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cdotc_sub");

    cblas_cdotc_sub(n, reinterpret_cast<const float *>(x), incX,
                       reinterpret_cast<const float *>(y), incY,
                       reinterpret_cast<float *>(&result));
}

// zdotu_sub
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const ComplexDouble *x, IndexType incX,
     const ComplexDouble *y, IndexType incY,
     ComplexDouble &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdotu_sub");

    cblas_zdotu_sub(n, reinterpret_cast<const double *>(x), incX,
                       reinterpret_cast<const double *>(y), incY,
                       reinterpret_cast<double *>(&result));
}

// zdotc_sub
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const ComplexDouble *x, IndexType incX,
    const ComplexDouble *y, IndexType incY,
    ComplexDouble &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdotc_sub");

    cblas_zdotc_sub(n, reinterpret_cast<const double *>(x), incX,
                       reinterpret_cast<const double *>(y), incY,
                       reinterpret_cast<double *>(&result));
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_DOT_TCC

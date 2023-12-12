/*
 *   Copyright (c) 2012, Klaus Pototzky
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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_DOT_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_DOT_TCC 1

#include <cstdio>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

#ifdef HAVE_CBLAS

template <typename IndexType>
void
dotu(IndexType n,
     const float *x, IndexType incX,
     const std::complex<float> *y, IndexType incY,
     std::complex<float> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cdotu [extension] [real,complex]");

    float real_result, imag_result;
    const float *yr = reinterpret_cast<const float *>(y);
    const float *yi = reinterpret_cast<const float *>(y) + 1;

    real_result = cblas_sdot(n, x, incX, yr, 2*incY);
    imag_result = cblas_sdot(n, x, incX, yi, 2*incY);
    result = std::complex<float>(real_result, imag_result);
}

template <typename IndexType>
void
dotu(IndexType n,
     const std::complex<float> *x, IndexType incX,
     const float *y, IndexType incY,
     std::complex<float> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cdotu [extension] [complex,real]");

    dotu(n, y, incY, x, incX, result);
}


template <typename IndexType>
void
dotu(IndexType n,
     const double *x, IndexType incX,
     const std::complex<double> *y, IndexType incY,
     std::complex<double> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdotu [extension] [real,complex]");

    double real_result, imag_result;
    const double *yr = reinterpret_cast<const double *>(y);
    const double *yi = reinterpret_cast<const double *>(y) + 1;

    real_result = cblas_ddot(n, x, incX, yr, 2*incY);
    imag_result = cblas_ddot(n, x, incX, yi, 2*incY);
    result = std::complex<double>(real_result, imag_result);
}

template <typename IndexType>
void
dotu(IndexType n,
     const std::complex<double> *x, IndexType incX,
     const double *y, IndexType incY,
     std::complex<double> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdotu [extension] [complex,real]");

    dotu(n, y, incY, x, incX, result);
}

template <typename IndexType>
void
dot(IndexType n,
    const float *x, IndexType incX,
    const std::complex<float> *y, IndexType incY,
    std::complex<float> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cdot [extension] [real,complex]");

    dotu(n, x, incX, y, incY, result);

}

template <typename IndexType>
void
dot(IndexType n,
    const std::complex<float> *x, IndexType incX,
    const float *y, IndexType incY,
    std::complex<float> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cdot [extension] [complex,real]");

    float real_result, imag_result;
    const float *xr = reinterpret_cast<const float *>(x);
    const float *xi = reinterpret_cast<const float *>(x) + 1;

    real_result = cblas_sdot(n, y, incY, xr, 2*incX);
    imag_result = cblas_sdot(n, y, incY, xi, 2*incX);
    result = std::complex<float>(real_result, -imag_result);

}

template <typename IndexType>
void
dot(IndexType n,
    const double *x, IndexType incX,
    const std::complex<double> *y, IndexType incY,
    std::complex<double> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdot [extension] [real, complex]");

    dotu(n, x, incX, y, incY, result);

}

template <typename IndexType>
void
dot(IndexType n,
    const std::complex<double> *x, IndexType incX,
    const double *y, IndexType incY,
    std::complex<double> &result)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zdot [extension] [complex, real]");

    double real_result, imag_result;
    const double *xr = reinterpret_cast<const double *>(x);
    const double *xi = reinterpret_cast<const double *>(x) + 1;

    real_result = cblas_ddot(n, y, incY, xr, 2*incX);
    imag_result = cblas_ddot(n, y, incY, xi, 2*incX);
    result = std::complex<double>(real_result, -imag_result);

}

#endif
} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_DOT_TCC

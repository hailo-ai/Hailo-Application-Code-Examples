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

#ifndef CXXBLAS_LEVEL2EXTENSIONS_GEMV_TCC
#define CXXBLAS_LEVEL2EXTENSIONS_GEMV_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {


template <typename IndexType, typename ALPHA, typename MA, typename VX,
          typename BETA, typename VY>
void
gemv(StorageOrder order, Transpose transA, Transpose conjX,
     IndexType m, IndexType n,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const VX *x, IndexType incX,
     const BETA &beta,
     VY *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv_generic (extension)");

    if ((m==0) || (n==0)) {
        return;
    }
    gemv_generic(order, transA, conjX, m, n,
                 alpha, A, ldA, x, incX,
                 beta, y, incY);
}

#ifdef HAVE_CBLAS

template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const float &alpha,
     const float *A, IndexType ldA,
     const float *x, IndexType incX,
     const float &beta,
     std::complex<float> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         x, incX, float(1), reinterpret_cast<float *>(y), 2*incY);

}

template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const float &alpha,
     const float *A, IndexType ldA,
     const float *x, IndexType incX,
     const std::complex<float> &beta,
     std::complex<float> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         x, incX, float(1), reinterpret_cast<float *>(y), 2*incY);
}


template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const float &alpha,
     const float *A, IndexType ldA,
     const std::complex<float> *x, IndexType incX,
     const float &beta,
     std::complex<float> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const float *>(x), 2*incX, float(1),
         reinterpret_cast<float *>(y), 2*incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const float *>(x)+1, 2*incX, float(1),
         reinterpret_cast<float *>(y)+1, 2*incY);

}


template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const float &alpha,
     const float *A, IndexType ldA,
     const std::complex<float> *x, IndexType incX,
     const std::complex<float> &beta,
     std::complex<float> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const float *>(x), 2*incX, float(1),
         reinterpret_cast<float *>(y), 2*incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const float *>(x)+1, 2*incX, float(1),
         reinterpret_cast<float *>(y)+1, 2*incY);

}

template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const double &alpha,
     const double *A, IndexType ldA,
     const double *x, IndexType incX,
     const double &beta,
     std::complex<double> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         x, incX, double(1), reinterpret_cast<double *>(y), 2*incY);

}

template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const double &alpha,
     const double *A, IndexType ldA,
     const double *x, IndexType incX,
     const std::complex<double> &beta,
     std::complex<double> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         x, incX, double(1), reinterpret_cast<double *>(y), 2*incY);
}


template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const double &alpha,
     const double *A, IndexType ldA,
     const std::complex<double> *x, IndexType incX,
     const double &beta,
     std::complex<double> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const double *>(x), 2*incX, double(1),
         reinterpret_cast<double *>(y), 2*incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const double *>(x)+1, 2*incX, double(1),
         reinterpret_cast<double *>(y)+1, 2*incY);

}


template <typename IndexType>
void
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const double &alpha,
     const double *A, IndexType ldA,
     const std::complex<double> *x, IndexType incX,
     const std::complex<double> &beta,
     std::complex<double> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv (extension)");

    scal(n, beta, y, incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const double *>(x), 2*incX, double(1),
         reinterpret_cast<double *>(y), 2*incY);

    gemv(order, transA, m, n, alpha, A, ldA,
         reinterpret_cast<const double *>(x)+1, 2*incX, double(1),
         reinterpret_cast<double *>(y)+1, 2*incY);

}

#endif

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2EXTENSIONS_GEMV_TCC

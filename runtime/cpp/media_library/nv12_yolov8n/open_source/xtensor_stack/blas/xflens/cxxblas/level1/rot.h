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

#ifndef CXXBLAS_LEVEL1_ROT_H
#define CXXBLAS_LEVEL1_ROT_H 1

#include "xflens/cxxblas/drivers/drivers.h"
#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_ROT  1
#define HAVE_CXXBLAS_ROTG 1

namespace cxxblas {

template <typename IndexType, typename X, typename Y, typename T>
    void
    rot(IndexType n, X *x, IndexType incX, Y *y, IndexType incY, T c, T s);

template <typename A, typename B, typename T>
    void
    rotg(A &a, B &b, T &c, T &s);

template <typename TA, typename TB, typename T>
    void
    rotg(std::complex<TA> &a, std::complex<TB> &b, T &c, std::complex<T> &s);

/*
 *  Note: The following variant of function rot is based on

       SUBROUTINE ZROT( N, CX, INCX, CY, INCY, C, S )
 *
 *  -- LAPACK auxiliary routine (version 3.2) --
 *  -- LAPACK is a software package provided by Univ. of Tennessee,    --
 *  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
 *     November 2006
 */
template <typename IndexType, typename X, typename Y, typename T>
    void
    rot(IndexType n,
        std::complex<X> *x, IndexType incX,
        std::complex<Y> *y, IndexType incY,
        T c, const std::complex<T> &s);


#ifdef HAVE_CBLAS
// srot
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    rot(IndexType n,
        float *x, IndexType incX,
        float *y, IndexType incY,
        float c, float s);

// drot
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    rot(IndexType n,
        double *x, IndexType incX,
        double *y, IndexType incY,
        double c, double s);

// srotg
template <typename T>
    typename RestrictTo<IsSame<T, float>::value, void>::Type
    rotg(T &a, T &b, T &c, T &s);

// drotg
template <typename T>
    typename RestrictTo<IsSame<T, double>::value, void>::Type
    rotg(T &a, T &b, T &c, T &s);

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_ROT_H

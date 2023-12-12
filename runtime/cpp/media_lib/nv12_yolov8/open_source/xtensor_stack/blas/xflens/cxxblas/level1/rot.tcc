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

#ifndef CXXBLAS_LEVEL1_ROT_TCC
#define CXXBLAS_LEVEL1_ROT_TCC 1

#include <cmath>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename X, typename Y, typename T>
void
rot_generic(IndexType n, X *x, IndexType incX, Y *y, IndexType incY, T c, T s)
{
    CXXBLAS_DEBUG_OUT("rot_generic");

    for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
        X x_ =  c*x[iX] + s*y[iY];
        Y y_ = -s*x[iX] + c*y[iY];
        x[iX] = x_;
        y[iY] = y_;
    }
}

template <typename IndexType, typename X, typename Y, typename T>
void
rot(IndexType n, X *x, IndexType incX, Y *y, IndexType incY, T c, T s)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    rot_generic(n, x, incX, y, incY, c, s);
}

template <typename A, typename B, typename T>
void
rotg(A &a, B &b, T &c, T &s)
{
    CXXBLAS_DEBUG_OUT("rotg (generic)");

    using std::abs;
    using std::sqrt;

    A absA = abs(a);
    B absB = abs(b);

    T scale = absA + absB;
    T roe = (absA > absB) ? a : b;
    if (scale==0) {
        c = 1;
        s = 0;
        a = 0;
        b = 0;
        return;
    }
    A aScaled = absA / scale;
    B bScaled = absB / scale;
    T r = scale*sqrt(aScaled*aScaled + bScaled*bScaled);
    if (roe<0) {
        r = -r;
    }
    c = a / r;
    s = b / r;

    B z = 1;
    if (absA > absB) {
        z = s;
    }
    if ((absA < absB) && (c != 0)) {
        z = T(1)/c;
    }
    a = r;
    b = z;
}

template <typename TA, typename TB, typename T>
void
rotg(std::complex<TA> &a, std::complex<TB> &b, T &c, std::complex<T> &s)
{
    using std::abs;
    using std::sqrt;
    using std::pow;

    std::complex<T>  alpha;
    T                norm, scale;

    if (abs(a)==TA(0)) {
        c = 0;
        s = std::complex<T>(1,0);
        a = b;
    } else {
        scale = abs(a) + abs(b);
        norm  = scale*sqrt(pow(abs(a/scale),2) + pow(abs(b/scale),2));
        alpha = a / abs(a);
        c     = abs(a) / norm;
        s     = alpha*conjugate(b)/norm;
        a     = alpha*norm;
    }
}

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
rot_generic(IndexType n,
            std::complex<X> *x, IndexType incX,
            std::complex<Y> *y, IndexType incY,
            T c, const std::complex<T> &s)
{
    using std::conj;

    typedef std::complex<T>   CT;

    if (incX != IndexType(1) || incY != IndexType(1)) {
//
//      Code for unequal increments or equal increments not equal to 1
//
        for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
            const CT tmp = c*x[iX] + s*y[iY];
            y[iY]  = c*y[iY] - conj(s)*x[iX];
            x[iX]  = tmp;
        }
    } else {
//
//      Code for both increments equal to 1
//
        for (IndexType i=0; i<n; ++i) {
            const CT tmp = c*x[i] + s*y[i];
            y[i] = c*y[i] - conj(s)*x[i];
            x[i] = tmp;
        }
    }
}

template <typename IndexType, typename X, typename Y, typename T>
void
rot(IndexType n,
    std::complex<X> *x, IndexType incX,
    std::complex<Y> *y, IndexType incY,
    T c, const std::complex<T> &s)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    rot_generic(n, x, incX, y, incY, c, s);
}

#ifdef HAVE_CBLAS
// srot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
rot(IndexType n, float *x, IndexType incX, float *y, IndexType incY,
    float c, float s)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_srot");

    cblas_srot(n, x, incX, y, incY, c, s);
}

// drot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
rot(IndexType n, double *x, IndexType incX, double *y, IndexType incY,
    double c, double s)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_drot");

    cblas_drot(n, x, incX, y, incY, c, s);
}

// srotg
template <typename T>
typename RestrictTo<IsSame<T, float>::value, void>::Type
rotg(T &a, T &b, T &c, T &s)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_srotg");

    cblas_srotg(&a, &b, &c, &s);
}

// drotg
template <typename T>
typename RestrictTo<IsSame<T, double>::value, void>::Type
rotg(T &a, T &b, T &c, T &s)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_drotg");

    cblas_drotg(&a, &b, &c, &s);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_ROT_TCC

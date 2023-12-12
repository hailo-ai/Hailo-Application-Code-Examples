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

#ifndef CXXBLAS_LEVEL1_NRM2_TCC
#define CXXBLAS_LEVEL1_NRM2_TCC 1

#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename X, typename T>
void
nrm2_generic(IndexType n, const X *x, IndexType incX, T &norm)
{
    CXXBLAS_DEBUG_OUT("nrm2_generic");

    using std::abs;
    using cxxblas::pow;
    using std::sqrt;

    const T  Zero(0), One(1);

    if (n<1) {
        norm = Zero;
    } else if (n==1) {
        norm = abs(*x);
    } else {
        T scale = 0;
        T ssq = 1;
//      The following loop is equivalent to this call to the LAPACK
//      auxiliary routine:
//      CALL DLASSQ( N, X, INCX, SCALE, SSQ )
//
        for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
            if (x[iX]!=Zero) {
                T absXi = abs(x[iX]);
                if (scale<absXi) {
                    ssq = One + ssq * pow(scale/absXi, 2);
                    scale = absXi;
                } else {
                    ssq += pow(absXi/scale, 2);
                }
            }
        }
        norm = scale*sqrt(ssq);
    }
}

template <typename IndexType, typename X, typename T>
void
nrm2_generic(IndexType n, const std::complex<X> *x, IndexType incX, T &norm)
{
    CXXBLAS_DEBUG_OUT("nrm2_generic");

    using std::abs;
    using std::imag;
    using std::pow;
    using std::real;
    using std::sqrt;

    const T  Zero(0), One(1);

    if (n<1) {
        norm = Zero;
    } else if (n==1) {
        norm = abs(*x);
    } else {
        T scale = 0;
        T ssq = 1;
//      The following loop is equivalent to this call to the LAPACK
//      auxiliary routine:
//      CALL DLASSQ( N, X, INCX, SCALE, SSQ )
//
        for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
            if (real(x[iX]) != Zero) {
                T absXi = abs(real(x[iX]));
                if (scale<absXi) {
                    ssq = One + ssq * pow(scale/absXi, 2);
                    scale = absXi;
                } else {
                    ssq += pow(absXi/scale, 2);
                }
            }
            if (imag(x[iX]) != Zero) {
                T absXi = abs(imag(x[iX]));
                if (scale<absXi) {
                    ssq = One + ssq * pow(scale/absXi, 2);
                    scale = absXi;
                } else {
                    ssq += pow(absXi/scale, 2);
                }
            }
        }
        norm = scale*sqrt(ssq);
    }
}

template <typename IndexType, typename X, typename T>
void
nrm2(IndexType n, const X *x, IndexType incX, T &norm)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    nrm2_generic(n, x, incX, norm);
}

#ifdef HAVE_CBLAS

// snrm2
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
nrm2(IndexType n, const float *x, IndexType incX, float &norm)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_snrm2");

    norm = cblas_snrm2(n, x, incX);
}

// dnrm2
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
nrm2(IndexType n, const double *x, IndexType incX, double &norm)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dnrm2");

    norm = cblas_dnrm2(n, x, incX);
}

// scnrm2
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
nrm2(IndexType n, const ComplexFloat *x, IndexType incX, float &norm)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_scnrm2");

    norm = cblas_scnrm2(n, reinterpret_cast<const float *>(x), incX);
}

// dznrm2
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
nrm2(IndexType n, const ComplexDouble *x, IndexType incX, double &norm)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dznrm2");

    norm = cblas_dznrm2(n, reinterpret_cast<const double *>(x), incX);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_NRM2_TCC

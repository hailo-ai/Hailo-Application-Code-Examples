/*
 *   Copyright (c) 2014, Michael Lehn
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

#ifndef CXXBLAS_AUXILIARY_POW_TCC
#define CXXBLAS_AUXILIARY_POW_TCC 1

#include <cmath>
#include "xflens/cxxblas/auxiliary/complextrait.h"
#include "xflens/cxxblas/auxiliary/pow.h"
#ifdef WITH_MPFR
#include <external/real.hpp>
#endif

namespace cxxblas {

template <typename T>
typename RestrictTo<IsSame<T,int>::value,
         T>::Type
pow(const T &base, const T &exponent)
{
    ASSERT( exponent>=0 );
    if ( exponent==0 ) {
        return 1;
    } else if ( exponent==1 ) {
        return base;
    }
    int value = cxxblas::pow(base, exponent/2 );

    if ( exponent%2==0 ) {
        return value*value;
    }
    return base*value*value;
}

#ifdef WITH_MPFR
template <typename T>
typename RestrictTo<!IsSame<T,int>::value
                 && !IsComplex<T>::value
                 && !IsMpfrReal<T>::value,
         T>::Type
pow(const T &base, int exponent)
{
//
//  TODO: Make this more general and call an external Fortran routine
//        that computes 'pow(base, exponent)' for comparison
//
#   ifdef CHECK_CXXLAPACK
    if (exponent==2) {
        return base*base;
    }
#   endif
    return std::pow(base, T(exponent));
}
#else
template <typename T>
typename RestrictTo<!IsSame<T,int>::value
                 && !IsComplex<T>::value,
         T>::Type
pow(const T &base, int exponent)
{
//
//  TODO: Make this more general and call an external Fortran routine
//        that computes 'pow(base, exponent)' for comparison
//
#   ifdef CHECK_CXXLAPACK
    if (exponent==2) {
        return base*base;
    }
#   endif
    return std::pow(base, T(exponent));
}
#endif

template <typename T>
std::complex<T>
pow(const std::complex<T> &base, int exponent)
{
//
//  TODO: Make this more general and call an external Fortran routine
//        that computes 'pow(base, exponent)' for comparison
//
#   ifdef CHECK_CXXLAPACK
    if (exponent==2) {
        return base*base;
    }
#   endif
    return std::pow(base, T(exponent));
}

} // namespace cxxblas

#endif // CXXBLAS_AUXILIARY_POW_TCC

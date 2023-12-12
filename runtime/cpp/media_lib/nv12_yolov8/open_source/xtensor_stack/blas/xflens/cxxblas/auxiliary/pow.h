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

#ifndef CXXBLAS_AUXILIARY_POW_H
#define CXXBLAS_AUXILIARY_POW_H 1

#include "xflens/cxxblas/auxiliary/iscomplex.h"
#include "xflens/cxxblas/auxiliary/ismpfrreal.h"
#include "xflens/cxxblas/auxiliary/issame.h"
#include "xflens/cxxblas/auxiliary/restrictto.h"

#ifdef WITH_MPFR
#include <external/real.hpp>
#endif

namespace cxxblas {

template <typename T>
    typename RestrictTo<IsSame<T,int>::value,
             T>::Type
    pow(const T &base, const T &exponent);

#ifdef WITH_MPFR
template <typename T>
    typename RestrictTo<!IsSame<T,int>::value
                     && !IsComplex<T>::value
                     && !IsMpfrReal<T>::value,
             T>::Type
    pow(const T &base, int exponent);
#else
template <typename T>
    typename RestrictTo<!IsSame<T,int>::value
                     && !IsComplex<T>::value,
             T>::Type
    pow(const T &base, int exponent);
#endif

template <typename T>
    std::complex<T>
    pow(const std::complex<T> &base, int exponent);

} // namespace cxxblas

#endif // CXXBLAS_AUXILIARY_POW_H

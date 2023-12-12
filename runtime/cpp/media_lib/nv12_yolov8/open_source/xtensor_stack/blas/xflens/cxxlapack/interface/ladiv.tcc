/*
 *   Copyright (c) 2012, Michael Lehn, Klaus Pototzky
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

#ifndef CXXLAPACK_INTERFACE_LADIV_TCC
#define CXXLAPACK_INTERFACE_LADIV_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename Double>
typename RestrictTo<IsSame<double, Double>::value,
         void>::Type
ladiv(const Double          a,
      const Double          b,
      const Double          c,
      const Double          d,
      Double                &p,
      Double                &q)
{
    CXXLAPACK_DEBUG_OUT("dladiv");

    LAPACK_IMPL(dladiv)(&a,
                        &b,
                        &c,
                        &d,
                        &p,
                        &q);
}


template <typename Double>
typename RestrictTo<IsSame<double, Double>::value,
         std::complex<double> >::Type
ladiv(std::complex<Double>  x,
      std::complex<Double>  y)
{
    CXXLAPACK_DEBUG_OUT("zladiv");

    std::complex<double> z
            =  LAPACK_IMPL(zladiv)(reinterpret_cast<const double *>(&x),
                                   reinterpret_cast<const double *>(&y));

    return z;
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LADIV_TCC

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

#ifndef CXXLAPACK_INTERFACE_LAESY_TCC
#define CXXLAPACK_INTERFACE_LAESY_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename XFLENS_VOID>
void
laesy(std::complex<float >  a,
      std::complex<float >  b,
      std::complex<float >  c,
      std::complex<float >  &rt1,
      std::complex<float >  &rt2,
      std::complex<float >  &evscal,
      std::complex<float >  &cs1,
      std::complex<float >  &sn1)

{
    CXXLAPACK_DEBUG_OUT("claesy");

    LAPACK_IMPL(claesy)(reinterpret_cast<const float  *>(&a),
                        reinterpret_cast<const float  *>(&b),
                        reinterpret_cast<const float  *>(&c),
                        reinterpret_cast<float  *>(&rt1),
                        reinterpret_cast<float  *>(&rt2),
                        reinterpret_cast<float  *>(&evscal),
                        reinterpret_cast<float  *>(&cs1),
                        reinterpret_cast<float  *>(&sn1));

}

template <typename XFLENS_VOID>
void
laesy(std::complex<double>  a,
      std::complex<double>  b,
      std::complex<double>  c,
      std::complex<double>  &rt1,
      std::complex<double>  &rt2,
      std::complex<double>  &evscal,
      std::complex<double>  &cs1,
      std::complex<double>  &sn1)

{
    CXXLAPACK_DEBUG_OUT("zlaesy");

    LAPACK_IMPL(zlaesy)(reinterpret_cast<const double *>(&a),
                        reinterpret_cast<const double *>(&b),
                        reinterpret_cast<const double *>(&c),
                        reinterpret_cast<double *>(&rt1),
                        reinterpret_cast<double *>(&rt2),
                        reinterpret_cast<double *>(&evscal),
                        reinterpret_cast<double *>(&cs1),
                        reinterpret_cast<double *>(&sn1));

}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAESY_TCC

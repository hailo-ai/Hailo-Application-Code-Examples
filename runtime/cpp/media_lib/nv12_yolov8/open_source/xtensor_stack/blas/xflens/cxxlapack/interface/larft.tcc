/*
 *   Copyright (c) 2012, Michael Lehn
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

#ifndef CXXLAPACK_INTERFACE_LARFT_TCC
#define CXXLAPACK_INTERFACE_LARFT_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
larft(char          direct,
      char          storev,
      IndexType     n,
      IndexType     k,
      float         *V,
      IndexType     ldV,
      const float   *tau,
      float         *T,
      IndexType     ldT)
{
    CXXLAPACK_DEBUG_OUT("slarft");

    LAPACK_IMPL(slarft)(&direct,
                        &storev,
                        &n,
                        &k,
                        V,
                        &ldV,
                        tau,
                        T,
                        &ldT);
}


template <typename IndexType>
void
larft(char          direct,
      char          storev,
      IndexType     n,
      IndexType     k,
      double        *V,
      IndexType     ldV,
      const double  *tau,
      double        *T,
      IndexType     ldT)
{
    CXXLAPACK_DEBUG_OUT("dlarft");

    LAPACK_IMPL(dlarft)(&direct,
                        &storev,
                        &n,
                        &k,
                        V,
                        &ldV,
                        tau,
                        T,
                        &ldT);
}

template <typename IndexType>
void
larft(char                          direct,
      char                          storev,
      IndexType                     n,
      IndexType                     k,
      std::complex<float >          *V,
      IndexType                     ldV,
      const std::complex<float >    *tau,
      std::complex<float >          *T,
      IndexType                     ldT)
{
    CXXLAPACK_DEBUG_OUT("clarft");

    LAPACK_IMPL(clarft)(&direct,
                        &storev,
                        &n,
                        &k,
                        reinterpret_cast<float  *>(V),
                        &ldV,
                        reinterpret_cast<const float  *>(tau),
                        reinterpret_cast<float  *>(T),
                        &ldT);
}

template <typename IndexType>
void
larft(char                          direct,
      char                          storev,
      IndexType                     n,
      IndexType                     k,
      std::complex<double>          *V,
      IndexType                     ldV,
      const std::complex<double>    *tau,
      std::complex<double>          *T,
      IndexType                     ldT)
{
    CXXLAPACK_DEBUG_OUT("zlarft");

    LAPACK_IMPL(zlarft)(&direct,
                        &storev,
                        &n,
                        &k,
                        reinterpret_cast<double *>(V),
                        &ldV,
                        reinterpret_cast<const double *>(tau),
                        reinterpret_cast<double *>(T),
                        &ldT);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARFT_TCC

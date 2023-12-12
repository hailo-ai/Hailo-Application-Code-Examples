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

#ifndef CXXLAPACK_INTERFACE_LAQR1_TCC
#define CXXLAPACK_INTERFACE_LAQR1_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqr1(IndexType         n,
      const float       *H,
      IndexType         ldH,
      const float       &sr1,
      const float       &si1,
      const float       &sr2,
      const float       &si2,
      float             *v)
{
    CXXLAPACK_DEBUG_OUT("slaqr1");

    LAPACK_IMPL(slaqr1)(&n,
                        H,
                        &ldH,
                        &sr1,
                        &si1,
                        &sr2,
                        &si2,
                        v);
}


template <typename IndexType>
void
laqr1(IndexType         n,
      const double      *H,
      IndexType         ldH,
      const double      &sr1,
      const double      &si1,
      const double      &sr2,
      const double      &si2,
      double            *v)
{
    CXXLAPACK_DEBUG_OUT("dlaqr1");

    LAPACK_IMPL(dlaqr1)(&n,
                        H,
                        &ldH,
                        &sr1,
                        &si1,
                        &sr2,
                        &si2,
                        v);
}

template <typename IndexType>
void
laqr1(IndexType                     n,
      const std::complex<float>     *H,
      IndexType                     ldH,
      const std::complex<float>     &s1,
      const std::complex<float>     &s2,
      std::complex<float >          *v)
{
    CXXLAPACK_DEBUG_OUT("claqr1");

    LAPACK_IMPL(claqr1)(&n,
                        reinterpret_cast<const float  *>(H),
                        &ldH,
                        reinterpret_cast<const float  *>(&s1),
                        reinterpret_cast<const float  *>(&s2),
                        reinterpret_cast<float  *>(v));
}

template <typename IndexType>
void
laqr1(IndexType                     n,
      const std::complex<double>    *H,
      IndexType                     ldH,
      const std::complex<double>    &s1,
      const std::complex<double>    &s2,
      std::complex<double>          *v)
{
    CXXLAPACK_DEBUG_OUT("zlaqr1");

    LAPACK_IMPL(zlaqr1)(&n,
                        reinterpret_cast<const double *>(H),
                        &ldH,
                        reinterpret_cast<const double *>(&s1),
                        reinterpret_cast<const double *>(&s2),
                        reinterpret_cast<double *>(v));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQR1_TCC

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

#ifndef CXXLAPACK_INTERFACE_LAR2V_TCC
#define CXXLAPACK_INTERFACE_LAR2V_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
lar2v(IndexType             n,
      float                 *x,
      float                 *y,
      float                 *z,
      IndexType             incx,
      const float           *c,
      const float           *s,
      IndexType             incc)
{
    CXXLAPACK_DEBUG_OUT("slaq2v");

    IndexType info;
    LAPACK_IMPL(slar2v)(&n,
                        x,
                        y,
                        z,
                        &incx,
                        c,
                        s,
                        &incc);
}


template <typename IndexType>
void
lar2v(IndexType             n,
      double                *x,
      double                *y,
      double                *z,
      IndexType             incx,
      const double          *c,
      const double          *s,
      IndexType             incc)
{
    CXXLAPACK_DEBUG_OUT("dlaq2v");

    IndexType info;
    LAPACK_IMPL(dlar2v)(&n,
                        x,
                        y,
                        z,
                        &incx,
                        c,
                        s,
                        &incc);
}

template <typename IndexType>
void
lar2v(IndexType                      n,
      std::complex<float >          *x,
      std::complex<float >          *y,
      std::complex<float >          *z,
      IndexType                     incx,
      const float                   *c,
      const std::complex<float >    *s,
      IndexType                     incc)
{
    CXXLAPACK_DEBUG_OUT("claq2v");

    IndexType info;
    LAPACK_IMPL(clar2v)(&n,
                        reinterpret_cast<float  *>(x),
                        reinterpret_cast<float  *>(y),
                        reinterpret_cast<float  *>(z),
                        &incx,
                        c,
                        reinterpret_cast<const float  *>(s),
                        &incc);
}

template <typename IndexType>
void
lar2v(IndexType                      n,
      std::complex<double>          *x,
      std::complex<double>          *y,
      std::complex<double>          *z,
      IndexType                     incx,
      const double                  *c,
      const std::complex<double>    *s,
      IndexType                     incc)
{
    CXXLAPACK_DEBUG_OUT("zlaq2v");

    IndexType info;
    LAPACK_IMPL(zlar2v)(&n,
                        reinterpret_cast<double *>(x),
                        reinterpret_cast<double *>(y),
                        reinterpret_cast<double *>(z),
                        &incx,
                        c,
                        reinterpret_cast<const double *>(s),
                        &incc);
}


} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAR2V_TCC

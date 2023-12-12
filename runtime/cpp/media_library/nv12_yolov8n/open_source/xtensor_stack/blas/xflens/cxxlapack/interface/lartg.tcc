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

#ifndef CXXLAPACK_INTERFACE_LARTG_TCC
#define CXXLAPACK_INTERFACE_LARTG_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename XFLENS_VOID>
void
lartg(const float      &f,
      const float      &g,
      float            &cs,
      float            &sn,
      float            &r)
{
    CXXLAPACK_DEBUG_OUT("slartg");

    LAPACK_IMPL(slartg)(&f,
                        &g,
                        &cs,
                        &sn,
                        &r);
}


template <typename XFLENS_VOID>
void
lartg(const double     &f,
      const double     &g,
      double           &cs,
      double           &sn,
      double           &r)
{
    CXXLAPACK_DEBUG_OUT("dlartg");

    LAPACK_IMPL(dlartg)(&f,
                        &g,
                        &cs,
                        &sn,
                        &r);
}

template <typename XFLENS_VOID>
void
lartg(const std::complex<float >    &f,
      const std::complex<float >    &g,
      float                         &cs,
      std::complex<float >          &sn,
      std::complex<float >          &r)
{
    CXXLAPACK_DEBUG_OUT("clartg");

    LAPACK_IMPL(clartg)(reinterpret_cast<const float  *>(&f),
                        reinterpret_cast<const float  *>(&g),
                        &cs,
                        reinterpret_cast<float  *>(&sn),
                        reinterpret_cast<float  *>(&r));
}

template <typename XFLENS_VOID>
void
lartg(const std::complex<double>    &f,
      const std::complex<double>    &g,
      double                        &cs,
      std::complex<double>          &sn,
      std::complex<double>          &r)
{
    CXXLAPACK_DEBUG_OUT("zlartg");

    LAPACK_IMPL(zlartg)(reinterpret_cast<const double *>(&f),
                        reinterpret_cast<const double *>(&g),
                        &cs,
                        reinterpret_cast<double *>(&sn),
                        reinterpret_cast<double *>(&r));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARTG_TCC

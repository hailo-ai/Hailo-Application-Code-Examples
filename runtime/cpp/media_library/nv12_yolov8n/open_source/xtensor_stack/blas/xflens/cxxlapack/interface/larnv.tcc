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

#ifndef CXXLAPACK_INTERFACE_LARNV_TCC
#define CXXLAPACK_INTERFACE_LARNV_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
larnv(IndexType             idist,
      IndexType             iseed,
      IndexType             n,
      float                 *x)
{
    CXXLAPACK_DEBUG_OUT("slarnv");

    LAPACK_IMPL(slarnv)(&idist,
                        &iseed,
                        &n,
                        x);
}


template <typename IndexType>
void
larnv(IndexType             idist,
      IndexType             iseed,
      IndexType             n,
      double                *x)
{
    CXXLAPACK_DEBUG_OUT("dlarnv");

    LAPACK_IMPL(dlarnv)(&idist,
                        &iseed,
                        &n,
                        x);
}

template <typename IndexType>
void
larnv(IndexType             idist,
      IndexType             iseed,
      IndexType             n,
      std::complex<float >  *x)
{
    CXXLAPACK_DEBUG_OUT("clarnv");

    LAPACK_IMPL(clarnv)(&idist,
                        &iseed,
                        &n,
                        reinterpret_cast<float  *>(x));
}

template <typename IndexType>
void
larnv(IndexType             idist,
      IndexType             iseed,
      IndexType             n,
      std::complex<double>  *x)
{
    CXXLAPACK_DEBUG_OUT("zlarnv");

    LAPACK_IMPL(zlarnv)(&idist,
                        &iseed,
                        &n,
                        reinterpret_cast<double *>(x));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARNV_TCC

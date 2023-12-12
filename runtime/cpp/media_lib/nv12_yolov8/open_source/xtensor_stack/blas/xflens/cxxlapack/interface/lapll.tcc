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

#ifndef CXXLAPACK_INTERFACE_LAPLL_TCC
#define CXXLAPACK_INTERFACE_LAPLL_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
lapll(IndexType             n,
      double                *x,
      IndexType             incx,
      float                 *y,
      float                 incy,
      float                 &ssmin)
{
    CXXLAPACK_DEBUG_OUT("slapll");

    LAPACK_IMPL(slapll)(&n,
                        x,
                        &incx,
                        y,
                        &incy,
                        &ssmin);
}

template <typename IndexType>
void
lapll(IndexType             n,
      double                *x,
      IndexType             incx,
      double                *y,
      double                incy,
      double                &ssmin)
{
    CXXLAPACK_DEBUG_OUT("dlapll");

    LAPACK_IMPL(dlapll)(&n,
                        x,
                        &incx,
                        y,
                        &incy,
                        &ssmin);
}

template <typename IndexType>
void
lapll(IndexType             n,
      std::complex<float >  *x,
      IndexType             incx,
      std::complex<float >  *y,
      float                 incy,
      float                 &ssmin)
{
    CXXLAPACK_DEBUG_OUT("clapll");

    LAPACK_IMPL(clapll)(&n,
                        reinterpret_cast<float  *>(x),
                        &incx,
                        reinterpret_cast<float  *>(y),
                        &incy,
                        &ssmin);
}

template <typename IndexType>
void
lapll(IndexType             n,
      std::complex<double>  *x,
      IndexType             incx,
      std::complex<double>  *y,
      double                incy,
      double                &ssmin)
{
    CXXLAPACK_DEBUG_OUT("zlapll");

    LAPACK_IMPL(zlapll)(&n,
                        reinterpret_cast<double *>(x),
                        &incx,
                        reinterpret_cast<double *>(y),
                        &incy,
                        &ssmin);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_DUMMY_TCC

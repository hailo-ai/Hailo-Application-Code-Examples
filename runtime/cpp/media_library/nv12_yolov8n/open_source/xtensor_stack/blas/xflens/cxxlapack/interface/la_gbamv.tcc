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

#ifndef CXXLAPACK_INTERFACE_LA_GBAMV_TCC
#define CXXLAPACK_INTERFACE_LA_GBAMV_TCC

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
la_gbamv(IndexType             trans,
         IndexType             m,
         IndexType             n,
         IndexType             kl,
         IndexType             ku,
         float                 alpha,
         const float           *AB,
         IndexType             ldAB,
         const float           *x,
         IndexType             incx,
         float                 beta,
         float                 *y,
         IndexType             incy)
{
    CXXLAPACK_DEBUG_OUT("sla_gbamv");

    LAPACK_IMPL(sla_gbamv)(&trans,
                           &m,
                           &n,
                           &kl,
                           &ku,
                           &alpha,
                           AB,
                           &ldAB,
                           x,
                           &incx,
                           &beta,
                           y,
                           &incy);
}

template <typename IndexType>
void
la_gbamv(IndexType             trans,
         IndexType             m,
         IndexType             n,
         IndexType             kl,
         IndexType             ku,
         double                alpha,
         const double          *AB,
         IndexType             ldAB,
         const double          *x,
         IndexType             incx,
         double                beta,
         double                *y,
         IndexType             incy)
{
    CXXLAPACK_DEBUG_OUT("dla_gbamv");

    LAPACK_IMPL(dla_gbamv)(&trans,
                           &m,
                           &n,
                           &kl,
                           &ku,
                           &alpha,
                           AB,
                           &ldAB,
                           x,
                           &incx,
                           &beta,
                           y,
                           &incy);
}

template <typename IndexType>
void
la_gbamv(IndexType                   trans,
         IndexType                   m,
         IndexType                   n,
         IndexType                   kl,
         IndexType                   ku,
         float                       alpha,
         const std::complex<float >  *AB,
         IndexType                   ldAB,
         const std::complex<float >  *x,
         IndexType                   incx,
         float                       beta,
         float                       *y,
         IndexType                   incy)
{
    CXXLAPACK_DEBUG_OUT("cla_gbamv");

    LAPACK_IMPL(cla_gbamv)(&trans,
                           &m,
                           &n,
                           &kl,
                           &ku,
                           &alpha,
                           reinterpret_cast<const float  *>(AB),
                           &ldAB,
                           reinterpret_cast<const float  *>(x),
                           &incx,
                           &beta,
                           y,
                           &incy);
}

template <typename IndexType>
void
la_gbamv(IndexType                   trans,
         IndexType                   m,
         IndexType                   n,
         IndexType                   kl,
         IndexType                   ku,
         double                      alpha,
         const std::complex<double>  *AB,
         IndexType                   ldAB,
         const std::complex<double>  *x,
         IndexType                   incx,
         double                      beta,
         double                      *y,
         IndexType                   incy)
{
    CXXLAPACK_DEBUG_OUT("zla_gbamv");

    LAPACK_IMPL(zla_gbamv)(&trans,
                           &m,
                           &n,
                           &kl,
                           &ku,
                           &alpha,
                           reinterpret_cast<const double *>(AB),
                           &ldAB,
                           reinterpret_cast<const double *>(x),
                           &incx,
                           &beta,
                           y,
                           &incy);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LA_GBAMV_TCC

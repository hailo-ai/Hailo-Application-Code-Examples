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

#ifndef CXXLAPACK_INTERFACE_LARFGP_TCC
#define CXXLAPACK_INTERFACE_LARFGP_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
larfgp(IndexType     n,
       float         &alpha,
       float         *x,
       IndexType     incX,
       float         &tau)
{
    CXXLAPACK_DEBUG_OUT("slarfgp");

    LAPACK_IMPL(slarfgp)(&n,
                         &alpha,
                         x,
                         &incX,
                         &tau);
}


template <typename IndexType>
void
larfgp(IndexType     n,
       double        &alpha,
       double        *x,
       IndexType     incX,
       double        &tau)
{
    CXXLAPACK_DEBUG_OUT("dlarfgp");

    LAPACK_IMPL(dlarfgp)(&n,
                         &alpha,
                         x,
                         &incX,
                         &tau);
}

template <typename IndexType>
void
larfgp(IndexType               n,
       std::complex<float >    &alpha,
       std::complex<float >    *x,
       IndexType               incX,
       std::complex<float >    &tau)
{
    CXXLAPACK_DEBUG_OUT("clarfgp");

    LAPACK_IMPL(clarfgp)(&n,
                         reinterpret_cast<float  *>(&alpha),
                         reinterpret_cast<float  *>(x),
                         &incX,
                         reinterpret_cast<float  *>(&tau));
}

template <typename IndexType>
void
larfgp(IndexType               n,
       std::complex<double>    &alpha,
       std::complex<double>    *x,
       IndexType               incX,
       std::complex<double>    &tau)
{
    CXXLAPACK_DEBUG_OUT("zlarfgp");

    LAPACK_IMPL(zlarfgp)(&n,
                         reinterpret_cast<double *>(&alpha),
                         reinterpret_cast<double *>(x),
                         &incX,
                         reinterpret_cast<double *>(&tau));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARFGP_TCC

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

#ifndef CXXLAPACK_INTERFACE_LAQPS_TCC
#define CXXLAPACK_INTERFACE_LAQPS_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqps(IndexType     m,
      IndexType     n,
      IndexType     offset,
      IndexType     nb,
      IndexType     &kb,
      float         *A,
      IndexType     ldA,
      IndexType     *jPvt,
      float         *tau,
      float         *vn1,
      float         *vn2,
      float         *auxv,
      float         *F,
      IndexType     ldF)
{
    CXXLAPACK_DEBUG_OUT("slaqps");

    LAPACK_IMPL(slaqps)(&m,
                        &n,
                        &offset,
                        &nb,
                        &kb,
                        A,
                        &ldA,
                        jPvt,
                        tau,
                        vn1,
                        vn2,
                        auxv,
                        F,
                        &ldF);
}


template <typename IndexType>
void
laqps(IndexType     m,
      IndexType     n,
      IndexType     offset,
      IndexType     nb,
      IndexType     &kb,
      double        *A,
      IndexType     ldA,
      IndexType     *jPvt,
      double        *tau,
      double        *vn1,
      double        *vn2,
      double        *auxv,
      double        *F,
      IndexType     ldF)
{
    CXXLAPACK_DEBUG_OUT("dlaqps");

    LAPACK_IMPL(dlaqps)(&m,
                        &n,
                        &offset,
                        &nb,
                        &kb,
                        A,
                        &ldA,
                        jPvt,
                        tau,
                        vn1,
                        vn2,
                        auxv,
                        F,
                        &ldF);
}

template <typename IndexType>
void
laqps(IndexType             m,
      IndexType             n,
      IndexType             offset,
      IndexType             nb,
      IndexType             &kb,
      std::complex<float >  *A,
      IndexType             ldA,
      IndexType             *jPvt,
      std::complex<float >  *tau,
      float                 *vn1,
      float                 *vn2,
      std::complex<float >  *auxv,
      std::complex<float >  *F,
      IndexType             ldF)
{
    CXXLAPACK_DEBUG_OUT("claqps");

    LAPACK_IMPL(claqps)(&m,
                        &n,
                        &offset,
                        &nb,
                        &kb,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        jPvt,
                        reinterpret_cast<float  *>(tau),
                        vn1,
                        vn2,
                        reinterpret_cast<float  *>(auxv),
                        reinterpret_cast<float  *>(F),
                        &ldF);
}

template <typename IndexType>
void
laqps(IndexType             m,
      IndexType             n,
      IndexType             offset,
      IndexType             nb,
      IndexType             &kb,
      std::complex<double>  *A,
      IndexType             ldA,
      IndexType             *jPvt,
      std::complex<double>  *tau,
      double                *vn1,
      double                *vn2,
      std::complex<double>  *auxv,
      std::complex<double>  *F,
      IndexType             ldF)
{
    CXXLAPACK_DEBUG_OUT("zlaqps");

    LAPACK_IMPL(zlaqps)(&m,
                        &n,
                        &offset,
                        &nb,
                        &kb,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        jPvt,
                        reinterpret_cast<double *>(tau),
                        vn1,
                        vn2,
                        reinterpret_cast<double *>(auxv),
                        reinterpret_cast<double *>(F),
                        &ldF);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQPS_TCC

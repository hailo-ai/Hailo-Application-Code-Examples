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

#ifndef CXXLAPACK_INTERFACE_LARZB_TCC
#define CXXLAPACK_INTERFACE_LARZB_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
larzb(char          side,
      char          trans,
      char          direct,
      char          storeV,
      IndexType     m,
      IndexType     n,
      IndexType     k,
      IndexType     l,
      const float   *V,
      IndexType     ldV,
      const float   *T,
      IndexType     ldT,
      float         *C,
      IndexType     ldC,
      float         *work,
      IndexType     ldWork)
{
    CXXLAPACK_DEBUG_OUT("slarzb");

    LAPACK_IMPL(slarzb)(&side,
                        &trans,
                        &direct,
                        &storeV,
                        &m,
                        &n,
                        &k,
                        &l,
                        V,
                        &ldV,
                        T,
                        &ldT,
                        C,
                        &ldC,
                        work,
                        &ldWork);
}


template <typename IndexType>
void
larzb(char          side,
      char          trans,
      char          direct,
      char          storeV,
      IndexType     m,
      IndexType     n,
      IndexType     k,
      IndexType     l,
      const double  *V,
      IndexType     ldV,
      const double  *T,
      IndexType     ldT,
      double        *C,
      IndexType     ldC,
      double        *work,
      IndexType     ldWork)
{
    CXXLAPACK_DEBUG_OUT("dlarzb");

    LAPACK_IMPL(dlarzb)(&side,
                        &trans,
                        &direct,
                        &storeV,
                        &m,
                        &n,
                        &k,
                        &l,
                        V,
                        &ldV,
                        T,
                        &ldT,
                        C,
                        &ldC,
                        work,
                        &ldWork);
}

template <typename IndexType>
void
larzb(char                        side,
      char                        trans,
      char                        direct,
      char                        storeV,
      IndexType                   m,
      IndexType                   n,
      IndexType                   k,
      IndexType                   l,
      const std::complex<float >  *V,
      IndexType                   ldV,
      const std::complex<float >  *T,
      IndexType                   ldT,
      std::complex<float >        *C,
      IndexType                   ldC,
      std::complex<float >        *work,
      IndexType                   ldWork)
{
    CXXLAPACK_DEBUG_OUT("clarzb");

    LAPACK_IMPL(clarzb)(&side,
                        &trans,
                        &direct,
                        &storeV,
                        &m,
                        &n,
                        &k,
                        &l,
                        reinterpret_cast<const float  *>(V),
                        &ldV,
                        reinterpret_cast<const float  *>(T),
                        &ldT,
                        reinterpret_cast<float  *>(C),
                        &ldC,
                        reinterpret_cast<float  *>(work),
                        &ldWork);
}

template <typename IndexType>
void
larzb(char                        side,
      char                        trans,
      char                        direct,
      char                        storeV,
      IndexType                   m,
      IndexType                   n,
      IndexType                   k,
      IndexType                   l,
      const std::complex<double>  *V,
      IndexType                   ldV,
      const std::complex<double>  *T,
      IndexType                   ldT,
      std::complex<double>        *C,
      IndexType                   ldC,
      std::complex<double>        *work,
      IndexType                   ldWork)
{
    CXXLAPACK_DEBUG_OUT("zlarzb");

    LAPACK_IMPL(zlarzb)(&side,
                        &trans,
                        &direct,
                        &storeV,
                        &m,
                        &n,
                        &k,
                        &l,
                        reinterpret_cast<const double *>(V),
                        &ldV,
                        reinterpret_cast<const double *>(T),
                        &ldT,
                        reinterpret_cast<double *>(C),
                        &ldC,
                        reinterpret_cast<double *>(work),
                        &ldWork);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARZB_TCC 1

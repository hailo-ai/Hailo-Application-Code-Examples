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

#ifndef CXXLAPACK_INTERFACE_LATZM_TCC
#define CXXLAPACK_INTERFACE_LATZM_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
latzm(char                  side,
      IndexType             m,
      IndexType             n,
      const float           *v,
      IndexType             incv,
      float                 tau,
      float                 *C1,
      float                 *C2,
      IndexType             ldC,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("slatzm");

    LAPACK_IMPL(slatzm)(&side,
                        &m,
                        &n,
                        v,
                        &incv,
                        &tau,
                        C1,
                        C2,
                        &ldC,
                        work);
}


template <typename IndexType>
void
latzm(char                  side,
      IndexType             m,
      IndexType             n,
      const double          *v,
      IndexType             incv,
      double                tau,
      double                *C1,
      double                *C2,
      IndexType             ldC,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dlatzm");

    LAPACK_IMPL(dlatzm)(&side,
                        &m,
                        &n,
                        v,
                        &incv,
                        &tau,
                        C1,
                        C2,
                        &ldC,
                        work);
}

template <typename IndexType>
void
latzm(char                       side,
      IndexType                  m,
      IndexType                  n,
      const std::complex<float > *v,
      IndexType                  incv,
      std::complex<float >       tau,
      std::complex<float >       *C1,
      std::complex<float >       *C2,
      IndexType                  ldC,
      std::complex<float >       *work)
{
    CXXLAPACK_DEBUG_OUT("clatzm");

    LAPACK_IMPL(clatzm)(&side,
                        &m,
                        &n,
                        reinterpret_cast<const float  *>(v),
                        &incv,
                        reinterpret_cast<const float  *>(&tau),
                        reinterpret_cast<float  *>(C1),
                        reinterpret_cast<float  *>(C2),
                        &ldC,
                        reinterpret_cast<float  *>(work));
}

template <typename IndexType>
void
latzm(char                       side,
      IndexType                  m,
      IndexType                  n,
      const std::complex<double> *v,
      IndexType                  incv,
      std::complex<double>       tau,
      std::complex<double>       *C1,
      std::complex<double>       *C2,
      IndexType                  ldC,
      std::complex<double>       *work)
{
    CXXLAPACK_DEBUG_OUT("zlatzm");

    LAPACK_IMPL(zlatzm)(&side,
                        &m,
                        &n,
                        reinterpret_cast<const double *>(v),
                        &incv,
                        reinterpret_cast<const double *>(&tau),
                        reinterpret_cast<double *>(C1),
                        reinterpret_cast<double *>(C2),
                        &ldC,
                        reinterpret_cast<double *>(work));
}


} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LATZM_TCC

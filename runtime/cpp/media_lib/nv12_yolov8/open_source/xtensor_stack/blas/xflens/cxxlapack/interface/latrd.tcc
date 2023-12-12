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

#ifndef CXXLAPACK_INTERFACE_LATRD_TCC
#define CXXLAPACK_INTERFACE_LATRD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
latrd(char                  uplo,
      IndexType             n,
      IndexType             nb,
      float                 *A,
      IndexType             ldA,
      float                 *e,
      float                 *tau,
      float                 *W,
      IndexType             ldW)
{
    CXXLAPACK_DEBUG_OUT("slatrd");

    LAPACK_IMPL(slatrd)(&uplo,
                        &n,
                        &nb,
                        A,
                        &ldA,
                        e,
                        tau,
                        W,
                        &ldW);

}


template <typename IndexType>
IndexType
latrd(char                  uplo,
      IndexType             n,
      IndexType             nb,
      double                *A,
      IndexType             ldA,
      double                *e,
      double                *tau,
      double                *W,
      IndexType             ldW)
{
    CXXLAPACK_DEBUG_OUT("dlatrd");

    LAPACK_IMPL(dlatrd)(&uplo,
                        &n,
                        &nb,
                        A,
                        &ldA,
                        e,
                        tau,
                        W,
                        &ldW);

}

template <typename IndexType>
void
latrd(char                       uplo,
      IndexType                  n,
      IndexType                  nb,
      std::complex<float >       *A,
      IndexType                  ldA,
      float                      *e,
      std::complex<float >       *tau,
      std::complex<float >       *W,
      IndexType                  ldW)
{
    CXXLAPACK_DEBUG_OUT("clatrd");

    LAPACK_IMPL(clatrd)(&uplo,
                        &n,
                        &nb,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        e,
                        reinterpret_cast<float  *>(tau),
                        reinterpret_cast<float  *>(W),
                        &ldW);
}

template <typename IndexType>
void
latrd(char                       uplo,
      IndexType                  n,
      IndexType                  nb,
      std::complex<double>       *A,
      IndexType                  ldA,
      double                     *e,
      std::complex<double>       *tau,
      std::complex<double>       *W,
      IndexType                  ldW)
{
    CXXLAPACK_DEBUG_OUT("zlatrd");

    LAPACK_IMPL(zlatrd)(&uplo,
                        &n,
                        &nb,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        e,
                        reinterpret_cast<double *>(tau),
                        reinterpret_cast<double *>(W),
                        &ldW);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LATRD_TCC

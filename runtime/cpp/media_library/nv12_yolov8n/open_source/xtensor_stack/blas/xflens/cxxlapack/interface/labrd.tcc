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

#ifndef CXXLAPACK_INTERFACE_LABRD_TCC
#define CXXLAPACK_INTERFACE_LABRD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
labrd(IndexType             m,
      IndexType             n,
      IndexType             nb,
      float                 *A,
      IndexType             ldA,
      float                 *d,
      float                 *e,
      float                 *tauq,
      float                 *taup,
      float                 *X,
      IndexType             ldX,
      float                 *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("slabrd");

    IndexType info;
    LAPACK_IMPL(slabrd)(&m,
                        &n,
                        &nb,
                        A,
                        &ldA,
                        d,
                        e,
                        tauq,
                        taup,
                        X,
                        &ldX,
                        Y,
                        &ldY);
}

template <typename IndexType>
void
labrd(IndexType             m,
      IndexType             n,
      IndexType             nb,
      double                *A,
      IndexType             ldA,
      double                *d,
      double                *e,
      double                *tauq,
      double                *taup,
      double                *X,
      IndexType             ldX,
      double                *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("dlabrd");

    IndexType info;
    LAPACK_IMPL(dlabrd)(&m,
                        &n,
                        &nb,
                        A,
                        &ldA,
                        d,
                        e,
                        tauq,
                        taup,
                        X,
                        &ldX,
                        Y,
                        &ldY);
}

template <typename IndexType>
void
labrd(IndexType             m,
      IndexType             n,
      IndexType             nb,
      std::complex<float >  *A,
      IndexType             ldA,
      float                 *d,
      float                 *e,
      std::complex<float >  *tauq,
      std::complex<float >  *taup,
      std::complex<float >  *X,
      IndexType             ldX,
      std::complex<float >  *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("clabrd");

    IndexType info;
    LAPACK_IMPL(clabrd)(&m,
                        &n,
                        &nb,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        d,
                        e,
                        reinterpret_cast<float  *>(tauq),
                        reinterpret_cast<float  *>(taup),
                        reinterpret_cast<float  *>(X),
                        &ldX,
                        reinterpret_cast<float  *>(Y),
                        &ldY);
}

template <typename IndexType>
void
labrd(IndexType             m,
      IndexType             n,
      IndexType             nb,
      std::complex<double>  *A,
      IndexType             ldA,
      double                *d,
      double                *e,
      std::complex<double>  *tauq,
      std::complex<double>  *taup,
      std::complex<double>  *X,
      IndexType             ldX,
      std::complex<double>  *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("zlabrd");

    IndexType info;
    LAPACK_IMPL(zlabrd)(&m,
                        &n,
                        &nb,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        d,
                        e,
                        reinterpret_cast<double *>(tauq),
                        reinterpret_cast<double *>(taup),
                        reinterpret_cast<double *>(X),
                        &ldX,
                        reinterpret_cast<double *>(Y),
                        &ldY);
}


} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LABRD_TCC

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

#ifndef CXXLAPACK_INTERFACE_GELSY_TCC
#define CXXLAPACK_INTERFACE_GELSY_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
gelsy(IndexType    m,
      IndexType    n,
      IndexType    nRhs,
      float        *A,
      IndexType    ldA,
      float        *B,
      IndexType    ldB,
      IndexType    *jPiv,
      float        rCond,
      IndexType    &rank,
      float        *work,
      IndexType    lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgelsy");
    LAPACK_IMPL(sgelsy)(&m,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        jPiv,
                        &rCond,
                        &rank,
                        work,
                        &lWork,
                        &info);

#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
}

template <typename IndexType>
void
gelsy(IndexType    m,
      IndexType    n,
      IndexType    nRhs,
      double       *A,
      IndexType    ldA,
      double       *B,
      IndexType    ldB,
      IndexType    *jPiv,
      double       rCond,
      IndexType    &rank,
      double       *work,
      IndexType    lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgelsy");
    LAPACK_IMPL(dgelsy)(&m,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        jPiv,
                        &rCond,
                        &rank,
                        work,
                        &lWork,
                        &info);

#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
}

template <typename IndexType>
void
gelsy(IndexType             m,
      IndexType             n,
      IndexType             nRhs,
      std::complex<float >  *A,
      const IndexType       ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      IndexType             *jPiv,
      float                 rCond,
      IndexType             &rank,
      std::complex<float >  *work,
      IndexType             lWork,
      float                 *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgelsy");
    LAPACK_IMPL(cgelsy)(&m,
                        &n,
                        &nRhs,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        jPiv,
                        &rCond,
                        &rank,
                        reinterpret_cast<float  *>(work),
                        &lWork,
                        rWork,
                        &info);

#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
}

template <typename IndexType>
void
gelsy(IndexType             m,
      IndexType             n,
      IndexType             nRhs,
      std::complex<double>  *A,
      const IndexType       ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      IndexType             *jPiv,
      double                rCond,
      IndexType             &rank,
      std::complex<double>  *work,
      IndexType             lWork,
      double                *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgelsy");
    LAPACK_IMPL(zgelsy)(&m,
                        &n,
                        &nRhs,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        jPiv,
                        &rCond,
                        &rank,
                        reinterpret_cast<double *>(work),
                        &lWork,
                        rWork,
                        &info);

#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_GELSY_TCC

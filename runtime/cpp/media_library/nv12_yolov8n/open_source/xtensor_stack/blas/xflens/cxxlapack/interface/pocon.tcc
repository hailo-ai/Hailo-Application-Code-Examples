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

#ifndef CXXLAPACK_INTERFACE_POCON_TCC
#define CXXLAPACK_INTERFACE_POCON_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
pocon(char              upLo,
      IndexType         n,
      const float       *A,
      IndexType         ldA,
      const float       &normA,
      float             &rCond,
      float             *work,
      IndexType         *iWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("spocon");
    LAPACK_IMPL(spocon)(&upLo,
                        &n,
                        A,
                        &ldA,
                        &normA,
                        &rCond,
                        work,
                        iWork,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}


template <typename IndexType>
IndexType
pocon(char              upLo,
      IndexType         n,
      const double      *A,
      IndexType         ldA,
      const double      &normA,
      double            &rCond,
      double            *work,
      IndexType         *iWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dpocon");
    LAPACK_IMPL(dpocon)(&upLo,
                        &n,
                        A,
                        &ldA,
                        &normA,
                        &rCond,
                        work,
                        iWork,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
pocon(char                          upLo,
      IndexType                     n,
      const std::complex<float >    *A,
      IndexType                     ldA,
      const float                   &normA,
      float                         &rCond,
      std::complex<float >          *work,
      float                         *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cpocon");
    LAPACK_IMPL(cpocon)(&upLo,
                        &n,
                        reinterpret_cast<const float  *>(A),
                        &ldA,
                        &normA,
                        &rCond,
                        reinterpret_cast<float  *>(work),
                        rWork,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
pocon(char                          upLo,
      IndexType                     n,
      const std::complex<double>    *A,
      IndexType                     ldA,
      const double                  &normA,
      double                        &rCond,
      std::complex<double>          *work,
      double                        *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zpocon");
    LAPACK_IMPL(zpocon)(&upLo,
                        &n,
                        reinterpret_cast<const double *>(A),
                        &ldA,
                        &normA,
                        &rCond,
                        reinterpret_cast<double *>(work),
                        rWork,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}



} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_POCON_TCC

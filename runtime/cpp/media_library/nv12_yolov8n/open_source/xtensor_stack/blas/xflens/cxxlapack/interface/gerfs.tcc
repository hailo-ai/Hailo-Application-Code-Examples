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

#ifndef CXXLAPACK_INTERFACE_GERFS_TCC
#define CXXLAPACK_INTERFACE_GERFS_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gerfs(char              trans,
      IndexType         n,
      IndexType         nRhs,
      const float       *A,
      IndexType         ldA,
      const float       *AF,
      IndexType         ldAF,
      const IndexType   *iPiv,
      const float       *B,
      IndexType         ldB,
      float             *X,
      IndexType         ldX,
      float             *fErr,
      float             *bErr,
      float             *work,
      IndexType         *iWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgerfs");
    LAPACK_IMPL(sgerfs)(&trans,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        AF,
                        &ldAF,
                        iPiv,
                        B,
                        &ldB,
                        X,
                        &ldX,
                        fErr,
                        bErr,
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
gerfs(char              trans,
      IndexType         n,
      IndexType         nRhs,
      const double      *A,
      IndexType         ldA,
      const double      *AF,
      IndexType         ldAF,
      const IndexType   *iPiv,
      const double      *B,
      IndexType         ldB,
      double            *X,
      IndexType         ldX,
      double            *fErr,
      double            *bErr,
      double            *work,
      IndexType         *iWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgerfs");
    LAPACK_IMPL(dgerfs)(&trans,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        AF,
                        &ldAF,
                        iPiv,
                        B,
                        &ldB,
                        X,
                        &ldX,
                        fErr,
                        bErr,
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
gerfs(char                          trans,
      IndexType                     n,
      IndexType                     nRhs,
      const std::complex<float >    *A,
      IndexType                     ldA,
      const std::complex<float >    *AF,
      IndexType                     ldAF,
      const IndexType               *iPiv,
      const std::complex<float >    *B,
      IndexType                     ldB,
      std::complex<float >          *X,
      IndexType                     ldX,
      float                         *fErr,
      float                         *bErr,
      std::complex<float >          *work,
      float                         *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgerfs");
    LAPACK_IMPL(cgerfs)(&trans,
                        &n,
                        &nRhs,
                        reinterpret_cast<const float  *>(A),
                        &ldA,
                        reinterpret_cast<const float  *>(AF),
                        &ldAF,
                        iPiv,
                        reinterpret_cast<const float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(X),
                        &ldX,
                        fErr,
                        bErr,
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
gerfs(char                          trans,
      IndexType                     n,
      IndexType                     nRhs,
      const std::complex<double>    *A,
      IndexType                     ldA,
      const std::complex<double>    *AF,
      IndexType                     ldAF,
      const IndexType               *iPiv,
      const std::complex<double>    *B,
      IndexType                     ldB,
      std::complex<double>          *X,
      IndexType                     ldX,
      double                        *fErr,
      double                        *bErr,
      std::complex<double>          *work,
      double                        *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgerfs");
    LAPACK_IMPL(zgerfs)(&trans,
                        &n,
                        &nRhs,
                        reinterpret_cast<const double *>(A),
                        &ldA,
                        reinterpret_cast<const double *>(AF),
                        &ldAF,
                        iPiv,
                        reinterpret_cast<const double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(X),
                        &ldX,
                        fErr,
                        bErr,
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

#endif // CXXLAPACK_INTERFACE_GERFS_TCC

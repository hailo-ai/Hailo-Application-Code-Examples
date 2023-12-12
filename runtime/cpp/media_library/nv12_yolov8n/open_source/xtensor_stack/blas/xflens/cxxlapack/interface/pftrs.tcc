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

#ifndef CXXLAPACK_INTERFACE_PFTRS_TCC
#define CXXLAPACK_INTERFACE_PFTRS_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
pftrs(char                  transr,
      char                  uplo,
      IndexType             n,
      IndexType             nRhs,
      const float           *A,
      float                 *B,
      IndexType             ldB)
{
    CXXLAPACK_DEBUG_OUT("spftrs");

    IndexType info;
    LAPACK_IMPL(spftrs)(&transr,
                        &uplo,
                        &n,
                        &nRhs,
                        A,
                        B,
                        &ldB,
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
pftrs(char                  transr,
      char                  uplo,
      IndexType             n,
      IndexType             nRhs,
      const double          *A,
      double                *B,
      IndexType             ldB)
{
    CXXLAPACK_DEBUG_OUT("dpftrs");

    IndexType info;
    LAPACK_IMPL(dpftrs)(&transr,
                        &uplo,
                        &n,
                        &nRhs,
                        A,
                        B,
                        &ldB,
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
pftrs(char                        transr,
      char                        uplo,
      IndexType                   n,
      IndexType                   nRhs,
      const std::complex<float >  *A,
      std::complex<float >        *B,
      IndexType                   ldB)
{
    CXXLAPACK_DEBUG_OUT("cpftrs");

    IndexType info;
    LAPACK_IMPL(cpftrs)(&transr,
                        &uplo,
                        &n,
                        &nRhs,
                        reinterpret_cast<const float  *>(A),
                        reinterpret_cast<float  *>(B),
                        &ldB,
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
pftrs(char                        transr,
      char                        uplo,
      IndexType                   n,
      IndexType                   nRhs,
      const std::complex<double>  *A,
      std::complex<double>        *B,
      IndexType                   ldB)
{
    CXXLAPACK_DEBUG_OUT("zpftrs");

    IndexType info;
    LAPACK_IMPL(zpftrs)(&transr,
                        &uplo,
                        &n,
                        &nRhs,
                        reinterpret_cast<const double *>(A),
                        reinterpret_cast<double *>(B),
                        &ldB,
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

#endif // CXXLAPACK_INTERFACE_PFTRS_TCC

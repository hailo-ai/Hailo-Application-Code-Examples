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

#ifndef CXXLAPACK_INTERFACE_TRSYL_TCC
#define CXXLAPACK_INTERFACE_TRSYL_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
trsyl(char          transA,
      char          transB,
      IndexType     sign,
      IndexType     m,
      IndexType     n,
      const float   *A,
      IndexType     ldA,
      const float   *B,
      IndexType     ldB,
      float         *C,
      IndexType     ldC,
      float         &scale)
{
    CXXLAPACK_DEBUG_OUT("strsyl");

    IndexType info;
    LAPACK_IMPL(strsyl)(&transA,
                        &transB,
                        &sign,
                        &m,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        C,
                        &ldC,
                        &scale,
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
trsyl(char          transA,
      char          transB,
      IndexType     sign,
      IndexType     m,
      IndexType     n,
      const double  *A,
      IndexType     ldA,
      const double  *B,
      IndexType     ldB,
      double        *C,
      IndexType     ldC,
      double        &scale)
{
    CXXLAPACK_DEBUG_OUT("dtrsyl");

    IndexType info;
    LAPACK_IMPL(dtrsyl)(&transA,
                        &transB,
                        &sign,
                        &m,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        C,
                        &ldC,
                        &scale,
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
trsyl(char                          transA,
      char                          transB,
      IndexType                     sign,
      IndexType                     m,
      IndexType                     n,
      const std::complex<float >    *A,
      const IndexType               ldA,
      const std::complex<float >    *B,
      const IndexType               ldB,
      std::complex<float >          *C,
      const IndexType               ldC,
      float                         &scale)
{
    CXXLAPACK_DEBUG_OUT("ctrsyl");

    IndexType info;
    LAPACK_IMPL(ctrsyl)(&transA,
                        &transB,
                        &sign,
                        &m,
                        &n,
                        reinterpret_cast<const float  *>(A),
                        &ldA,
                        reinterpret_cast<const float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(C),
                        &ldC,
                        &scale,
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
trsyl(char                          transA,
      char                          transB,
      IndexType                     sign,
      IndexType                     m,
      IndexType                     n,
      const std::complex<double>    *A,
      const IndexType               ldA,
      const std::complex<double>    *B,
      const IndexType               ldB,
      std::complex<double>          *C,
      const IndexType               ldC,
      double                        &scale)
{
    CXXLAPACK_DEBUG_OUT("ztrsyl");

    IndexType info;
    LAPACK_IMPL(ztrsyl)(&transA,
                        &transB,
                        &sign,
                        &m,
                        &n,
                        reinterpret_cast<const double *>(A),
                        &ldA,
                        reinterpret_cast<const double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(C),
                        &ldC,
                        &scale,
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

#endif // CXXLAPACK_INTERFACE_TRSYL_TCC

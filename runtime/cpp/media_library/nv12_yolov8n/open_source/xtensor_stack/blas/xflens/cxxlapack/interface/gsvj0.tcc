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

#ifndef CXXLAPACK_INTERFACE_GSVJ0_TCC
#define CXXLAPACK_INTERFACE_GSVJ0_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gsvj0(char          jobV,
      IndexType     m,
      IndexType     n,
      float         *A,
      IndexType     ldA,
      float         *D,
      float         *sva,
      IndexType     mv,
      float         *V,
      IndexType     ldV,
      const float   &eps,
      const float   &sfMin,
      const float   &tol,
      IndexType     nSweep,
      float         *work,
      IndexType     lWork)
{
    CXXLAPACK_DEBUG_OUT("sgsvj0");

    IndexType info;
    LAPACK_IMPL(sgsvj0)(&jobV,
                        &m,
                        &n,
                        A,
                        &ldA,
                        D,
                        sva,
                        &mv,
                        V,
                        &ldV,
                        &eps,
                        &sfMin,
                        &tol,
                        &nSweep,
                        work,
                        &lWork,
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
gsvj0(char          jobV,
      IndexType     m,
      IndexType     n,
      double        *A,
      IndexType     ldA,
      double        *D,
      double        *sva,
      IndexType     mv,
      double        *V,
      IndexType     ldV,
      const double  &eps,
      const double  &sfMin,
      const double  &tol,
      IndexType     nSweep,
      double        *work,
      IndexType     lWork)
{
    CXXLAPACK_DEBUG_OUT("dgsvj0");

    IndexType info;
    LAPACK_IMPL(dgsvj0)(&jobV,
                        &m,
                        &n,
                        A,
                        &ldA,
                        D,
                        sva,
                        &mv,
                        V,
                        &ldV,
                        &eps,
                        &sfMin,
                        &tol,
                        &nSweep,
                        work,
                        &lWork,
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

#endif // CXXLAPACK_INTERFACE_GSVJ0_TCC

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

#ifndef CXXLAPACK_INTERFACE_TREXC_TCC
#define CXXLAPACK_INTERFACE_TREXC_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
trexc(char          compQ,
      IndexType     n,
      float         *T,
      IndexType     ldT,
      float         *Q,
      IndexType     ldQ,
      IndexType     &iFirst,
      IndexType     &iLast,
      float         *work)
{
    CXXLAPACK_DEBUG_OUT("strexc");

    IndexType info;
    LAPACK_IMPL(strexc)(&compQ,
                        &n,
                        T,
                        &ldT,
                        Q,
                        &ldQ,
                        &iFirst,
                        &iLast,
                        work,
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
trexc(char          compQ,
      IndexType     n,
      double        *T,
      IndexType     ldT,
      double        *Q,
      IndexType     ldQ,
      IndexType     &iFirst,
      IndexType     &iLast,
      double        *work)
{
    CXXLAPACK_DEBUG_OUT("dtrexc");

    IndexType info;
    LAPACK_IMPL(dtrexc)(&compQ,
                        &n,
                        T,
                        &ldT,
                        Q,
                        &ldQ,
                        &iFirst,
                        &iLast,
                        work,
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
trexc(char                  compQ,
      IndexType             n,
      std::complex<float >  *T,
      IndexType             ldT,
      std::complex<float >  *Q,
      IndexType             ldQ,
      IndexType             iFirst,
      IndexType             iLast)
{
    CXXLAPACK_DEBUG_OUT("ctrexc");

    IndexType info;
    LAPACK_IMPL(ctrexc)(&compQ,
                        &n,
                        reinterpret_cast<float  *>(T),
                        &ldT,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        &iFirst,
                        &iLast,
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
trexc(char                  compQ,
      IndexType             n,
      std::complex<double>  *T,
      IndexType             ldT,
      std::complex<double>  *Q,
      IndexType             ldQ,
      IndexType             iFirst,
      IndexType             iLast)
{
    CXXLAPACK_DEBUG_OUT("ztrexc");

    IndexType info;
    LAPACK_IMPL(ztrexc)(&compQ,
                        &n,
                        reinterpret_cast<double *>(T),
                        &ldT,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        &iFirst,
                        &iLast,
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

#endif // CXXLAPACK_INTERFACE_TREXC_TCC

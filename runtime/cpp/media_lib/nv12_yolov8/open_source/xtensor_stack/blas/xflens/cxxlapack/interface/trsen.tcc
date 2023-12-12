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

#ifndef CXXLAPACK_INTERFACE_TRSEN_TCC
#define CXXLAPACK_INTERFACE_TRSEN_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
trsen(char              job,
      char              compQ,
      const IndexType   *select,
      IndexType         n,
      float             *T,
      IndexType         ldT,
      double            *Q,
      IndexType         ldQ,
      float             *wr,
      float             *wi,
      IndexType         &m,
      float             &s,
      float             &sep,
      float             *work,
      IndexType         lWork,
      IndexType         *iWork,
      IndexType         liWork)
{
    CXXLAPACK_DEBUG_OUT("strsen");

    IndexType info;
    LAPACK_IMPL(strsen)(&job,
                        &compQ,
                        select,
                        &n,
                        T,
                        &ldT,
                        Q,
                        &ldQ,
                        wr,
                        wi,
                        &m,
                        &s,
                        &sep,
                        work,
                        &lWork,
                        iWork,
                        &liWork,
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
trsen(char              job,
      char              compQ,
      const IndexType   *select,
      IndexType         n,
      double            *T,
      IndexType         ldT,
      double            *Q,
      IndexType         ldQ,
      double            *wr,
      double            *wi,
      IndexType         &m,
      double            &s,
      double            &sep,
      double            *work,
      IndexType         lWork,
      IndexType         *iWork,
      IndexType         liWork)
{
    CXXLAPACK_DEBUG_OUT("dtrsen");

    IndexType info;
    LAPACK_IMPL(dtrsen)(&job,
                        &compQ,
                        select,
                        &n,
                        T,
                        &ldT,
                        Q,
                        &ldQ,
                        wr,
                        wi,
                        &m,
                        &s,
                        &sep,
                        work,
                        &lWork,
                        iWork,
                        &liWork,
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
trsen(char                  job,
      char                  compQ,
      const IndexType       *select,
      IndexType             n,
      std::complex<float >  *T,
      const IndexType       ldT,
      std::complex<float >  *Q,
      const IndexType       ldQ,
      std::complex<float >  *w,
      IndexType             &m,
      float                 &s,
      float                 &sep,
      std::complex<float >  *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("ctrsen");

    IndexType info;
    LAPACK_IMPL(ctrsen)(&job,
                        &compQ,
                        *select,
                        &n,
                        reinterpret_cast<float  *>(T),
                        &ldT,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        reinterpret_cast<float  *>(w),
                        &m,
                        &s,
                        &sep,
                        reinterpret_cast<float  *>(work),
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
trsen(char                  job,
      char                  compQ,
      const IndexType       *select,
      IndexType             n,
      std::complex<double>  *T,
      const IndexType       ldT,
      std::complex<double>  *Q,
      const IndexType       ldQ,
      std::complex<double>  *w,
      IndexType             &m,
      double                &s,
      double                &sep,
      std::complex<double>  *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("ztrsen");

    IndexType info;
    LAPACK_IMPL(ztrsen)(&job,
                        &compQ,
                        *select,
                        &n,
                        reinterpret_cast<double *>(T),
                        &ldT,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        reinterpret_cast<double *>(w),
                        &m,
                        &s,
                        &sep,
                        reinterpret_cast<double *>(work),
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

#endif // CXXLAPACK_INTERFACE_TRSEN_TCC

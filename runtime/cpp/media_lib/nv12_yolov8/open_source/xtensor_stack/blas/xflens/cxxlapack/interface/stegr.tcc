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

#ifndef CXXLAPACK_INTERFACE_STEGR_TCC
#define CXXLAPACK_INTERFACE_STEGR_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
stegr(char                  jobz,
      char                  range,
      IndexType             n,
      float                 *d,
      float                 *e,
      float                 vl,
      float                 vu,
      IndexType             il,
      IndexType             iu,
      float                 abstol,
      IndexType             &m,
      float                 *w,
      float                 *Z,
      IndexType             ldZ,
      IndexType             *isuppz,
      float                 *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("sstegr");

    IndexType info;
    LAPACK_IMPL(sstegr)(&jobz,
                        &range,
                        &n,
                        d,
                        e,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        &abstol,
                        &m,
                        w,
                        Z,
                        &ldZ,
                        isuppz,
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
stegr(char                  jobz,
      char                  range,
      IndexType             n,
      double                *d,
      double                *e,
      double                vl,
      double                vu,
      IndexType             il,
      IndexType             iu,
      double                abstol,
      IndexType             &m,
      double                *w,
      double                *Z,
      IndexType             ldZ,
      IndexType             *isuppz,
      double                *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("dstegr");

    IndexType info;
    LAPACK_IMPL(dstegr)(&jobz,
                        &range,
                        &n,
                        d,
                        e,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        &abstol,
                        &m,
                        w,
                        Z,
                        &ldZ,
                        isuppz,
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
stegr(char                  jobz,
      char                  range,
      IndexType             n,
      float                 *d,
      float                 *e,
      float                 vl,
      float                 vu,
      IndexType             il,
      IndexType             iu,
      float                 abstol,
      IndexType             &m,
      float                 *w,
      std::complex<float >  *Z,
      IndexType             ldZ,
      IndexType             *isuppz,
      float                 *work,
      IndexType             lWork,
      IndexType             *iWork,
      IndexType             liWork)
{
    CXXLAPACK_DEBUG_OUT("cstegr");

    IndexType info;
    LAPACK_IMPL(cstegr)(&jobz,
                        &range,
                        &n,
                        d,
                        e,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        &abstol,
                        &m,
                        w,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
                        isuppz,
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
stegr(char                  jobz,
      char                  range,
      IndexType             n,
      double                *d,
      double                *e,
      double                vl,
      double                vu,
      IndexType             il,
      IndexType             iu,
      double                abstol,
      IndexType             &m,
      double                *w,
      std::complex<double>  *Z,
      IndexType             ldZ,
      IndexType             *isuppz,
      double                *work,
      IndexType             lWork,
      IndexType             *iWork,
      IndexType             liWork)
{
    CXXLAPACK_DEBUG_OUT("z‚stegr");

    IndexType info;
    LAPACK_IMPL(zstegr)(&jobz,
                        &range,
                        &n,
                        d,
                        e,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        &abstol,
                        &m,
                        w,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
                        isuppz,
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

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_STEGR_TCC

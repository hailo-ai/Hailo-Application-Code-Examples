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

#ifndef CXXLAPACK_INTERFACE_HBEVX_TCC
#define CXXLAPACK_INTERFACE_HBEVX_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
hbevx(char                  jobz,
      char                  range,
      char                  uplo,
      IndexType             n,
      IndexType             kd,
      std::complex<float >  *Ab,
      IndexType             ldAb,
      std::complex<float >  *Q,
      IndexType             ldQ,
      float                 vl,
      float                 vu,
      IndexType             il,
      IndexType             iu,
      float                 abstol,
      IndexType             &m,
      float                 *w,
      std::complex<float >  *Z,
      IndexType             ldZ,
      std::complex<float >  *work,
      float                 *rWork,
      IndexType             *ifail)
{
    CXXLAPACK_DEBUG_OUT("chbevx");

    IndexType info;
    LAPACK_IMPL(chbevx)(&jobz,
                        &range,
                        &uplo,
                        &n,
                        &kd,
                        reinterpret_cast<float  *>(Ab),
                        &ldAb,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        &abstol,
                        &m,
                        w,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
                        reinterpret_cast<float *>(work),
                        rWork,
                        ifail,
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
hbevx(char                  jobz,
      char                  range,
      char                  uplo,
      IndexType             n,
      IndexType             kd,
      std::complex<double>  *Ab,
      IndexType             ldAb,
      std::complex<double>  *Q,
      IndexType             ldQ,
      double                vl,
      double                vu,
      IndexType             il,
      IndexType             iu,
      double                abstol,
      IndexType             &m,
      double                *w,
      std::complex<double>  *Z,
      IndexType             ldZ,
      std::complex<double>  *work,
      double                *rWork,
      IndexType             *ifail)
{
    CXXLAPACK_DEBUG_OUT("zhbevx");

    IndexType info;
    LAPACK_IMPL(zhbevx)(&jobz,
                        &range,
                        &uplo,
                        &n,
                        &kd,
                        reinterpret_cast<double *>(Ab),
                        &ldAb,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        &abstol,
                        &m,
                        w,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
                        reinterpret_cast<double *>(work),
                        rWork,
                        ifail,
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

#endif // CXXLAPACK_INTERFACE_HBEVX_TCC

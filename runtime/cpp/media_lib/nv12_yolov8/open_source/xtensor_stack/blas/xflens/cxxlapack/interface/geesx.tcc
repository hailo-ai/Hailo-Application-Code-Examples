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

#ifndef CXXLAPACK_INTERFACE_GEESX_TCC
#define CXXLAPACK_INTERFACE_GEESX_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
geesx(char              jobVS,
      char              sort,
      IndexType         (*select)(const float *, const float  *),
      char              sense,
      IndexType         n,
      float             *A,
      IndexType         ldA,
      IndexType         &sDim,
      float             *wr,
      float             *wi,
      float             *VS,
      IndexType         ldVS,
      float             &rCondE,
      float             &rCondV,
      float             *work,
      IndexType         lWork,
      IndexType         *iWork,
      IndexType         liWork,
      IndexType         *bWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgeesx");
    LAPACK_IMPL(sgeesx)(&jobVS,
                        &sort,
                        select,
                        &sense,
                        &n,
                        A,
                        &ldA,
                        &sDim,
                        wr,
                        wi,
                        VS,
                        &ldVS,
                        &rCondE,
                        &rCondV,
                        work,
                        &lWork,
                        iWork,
                        &liWork,
                        bWork,
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
geesx(char              jobVS,
      char              sort,
      IndexType         (*select)(const double *, const double *),
      char              sense,
      IndexType         n,
      double            *A,
      IndexType         ldA,
      IndexType         &sDim,
      double            *wr,
      double            *wi,
      double            *VS,
      IndexType         ldVS,
      double            &rCondE,
      double            &rCondV,
      double            *work,
      IndexType         lWork,
      IndexType         *iWork,
      IndexType         liWork,
      IndexType         *bWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgeesx");
    LAPACK_IMPL(dgeesx)(&jobVS,
                        &sort,
                        select,
                        &sense,
                        &n,
                        A,
                        &ldA,
                        &sDim,
                        wr,
                        wi,
                        VS,
                        &ldVS,
                        &rCondE,
                        &rCondV,
                        work,
                        &lWork,
                        iWork,
                        &liWork,
                        bWork,
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
geesx(char                  jobVS,
      char                  sort,
      IndexType             (*select)(const std::complex<float > *),
      char                  sense,
      IndexType             n,
      std::complex<float >  *A,
      IndexType             ldA,
      IndexType             &sDim,
      std::complex<float >  *w,
      std::complex<float >  *VS,
      IndexType             ldVS,
      float                 &rCondE,
      float                 &rCondV,
      std::complex<float >  *work,
      IndexType             lWork,
      float                 *rWork,
      IndexType             *bWork)
{
    typedef IndexType (*LapackSelect)(const float  *);

    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgeesx");
    LAPACK_IMPL(cgeesx)(&jobVS,
                        &sort,
                        reinterpret_cast<LapackSelect>(select),
                        &sense,
                        &n,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        &sDim,
                        reinterpret_cast<float  *>(w),
                        reinterpret_cast<float  *>(VS),
                        &ldVS,
                        &rCondE,
                        &rCondV,
                        reinterpret_cast<float  *>(work),
                        &lWork,
                        rWork,
                        bWork,
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
geesx(char                  jobVS,
      char                  sort,
      IndexType             (*select)(const std::complex<double> *),
      char                  sense,
      IndexType             n,
      std::complex<double>  *A,
      IndexType             ldA,
      IndexType             &sDim,
      std::complex<double>  *w,
      std::complex<double>  *VS,
      IndexType             ldVS,
      double                &rCondE,
      double                &rCondV,
      std::complex<double>  *work,
      IndexType             lWork,
      double                *rWork,
      IndexType             *bWork)
{
    typedef IndexType (*LapackSelect)(const double *);

    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgeesx");
    LAPACK_IMPL(zgeesx)(&jobVS,
                        &sort,
                        reinterpret_cast<LapackSelect>(select),
                        &sense,
                        &n,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        &sDim,
                        reinterpret_cast<double *>(w),
                        reinterpret_cast<double *>(VS),
                        &ldVS,
                        &rCondE,
                        &rCondV,
                        reinterpret_cast<double *>(work),
                        &lWork,
                        rWork,
                        bWork,
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

#endif // CXXLAPACK_INTERFACE_GEESX_TCC

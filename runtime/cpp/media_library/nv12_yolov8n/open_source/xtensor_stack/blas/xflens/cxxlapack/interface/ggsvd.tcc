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

#ifndef CXXLAPACK_INTERFACE_GGSVD_TCC
#define CXXLAPACK_INTERFACE_GGSVD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
ggsvd(char                  jobu,
      char                  jobv,
      char                  jobq,
      IndexType             m,
      IndexType             n,
      IndexType             p,
      IndexType             &k,
      IndexType             &l,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      float                 *alpha,
      float                 *beta,
      float                 *U,
      IndexType             ldU,
      float                 *V,
      IndexType             ldV,
      float                 *Q,
      IndexType             ldQ,
      float                 *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("sggsvd");

    IndexType info;
    LAPACK_IMPL(sggsvd)(&jobu,
                        &jobv,
                        &jobq,
                        &m,
                        &n,
                        &p,
                        &k,
                        &l,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        alpha,
                        beta,
                        U,
                        &ldU,
                        V,
                        &ldV,
                        Q,
                        &ldQ,
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
ggsvd(char                  jobu,
      char                  jobv,
      char                  jobq,
      IndexType             m,
      IndexType             n,
      IndexType             p,
      IndexType             &k,
      IndexType             &l,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      double                *alpha,
      double                *beta,
      double                *U,
      IndexType             ldU,
      double                *V,
      IndexType             ldV,
      double                *Q,
      IndexType             ldQ,
      double                *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("dggsvd");

    IndexType info;
    LAPACK_IMPL(dggsvd)(&jobu,
                        &jobv,
                        &jobq,
                        &m,
                        &n,
                        &p,
                        &k,
                        &l,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        alpha,
                        beta,
                        U,
                        &ldU,
                        V,
                        &ldV,
                        Q,
                        &ldQ,
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
ggsvd(char                  jobu,
      char                  jobv,
      char                  jobq,
      IndexType             m,
      IndexType             n,
      IndexType             p,
      IndexType             &k,
      IndexType             &l,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      float                 *alpha,
      float                 *beta,
      std::complex<float >  *U,
      IndexType             ldU,
      std::complex<float >  *V,
      IndexType             ldV,
      std::complex<float >  *Q,
      IndexType             ldQ,
      std::complex<float >  *work,
      float                 *rWork,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("cggsvd");

    IndexType info;
    LAPACK_IMPL(cggsvd)(&jobu,
                        &jobv,
                        &jobq,
                        &m,
                        &n,
                        &p,
                        &k,
                        &l,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        alpha,
                        beta,
                        reinterpret_cast<float  *>(U),
                        &ldU,
                        reinterpret_cast<float  *>(V),
                        &ldV,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        reinterpret_cast<float  *>(work),
                        rWork,
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
ggsvd(char                  jobu,
      char                  jobv,
      char                  jobq,
      IndexType             m,
      IndexType             n,
      IndexType             p,
      IndexType             &k,
      IndexType             &l,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      double                *alpha,
      double                *beta,
      std::complex<double>  *U,
      IndexType             ldU,
      std::complex<double>  *V,
      IndexType             ldV,
      std::complex<double>  *Q,
      IndexType             ldQ,
      std::complex<double>  *work,
      double                *rWork,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("zggsvd");

    IndexType info;
    LAPACK_IMPL(zggsvd)(&jobu,
                        &jobv,
                        &jobq,
                        &m,
                        &n,
                        &p,
                        &k,
                        &l,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        alpha,
                        beta,
                        reinterpret_cast<double *>(U),
                        &ldU,
                        reinterpret_cast<double *>(V),
                        &ldV,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        reinterpret_cast<double *>(work),
                        rWork,
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


} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_GGSVD_TCC

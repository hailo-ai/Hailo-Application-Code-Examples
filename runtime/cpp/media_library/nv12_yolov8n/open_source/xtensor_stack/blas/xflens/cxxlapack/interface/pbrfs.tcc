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

#ifndef CXXLAPACK_INTERFACE_PBRFS_TCC
#define CXXLAPACK_INTERFACE_PBRFS_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
pbrfs(char                  uplo,
      IndexType             n,
      IndexType             kd,
      IndexType             nRhs,
      const float           *Ab,
      IndexType             ldAb,
      const float           *Afb,
      IndexType             ldAfb,
      const float           *B,
      IndexType             ldB,
      float                 *X,
      IndexType             ldX,
      float                 *ferr,
      float                 *berr,
      float                 *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("spbrfs");

    IndexType info;
    LAPACK_IMPL(spbrfs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        Ab,
                        &ldAb,
                        Afb,
                        &ldAfb,
                        B,
                        &ldB,
                        X,
                        &ldX,
                        ferr,
                        berr,
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
pbrfs(char                  uplo,
      IndexType             n,
      IndexType             kd,
      IndexType             nRhs,
      const double          *Ab,
      IndexType             ldAb,
      const double          *Afb,
      IndexType             ldAfb,
      const double          *B,
      IndexType             ldB,
      double                *X,
      IndexType             ldX,
      double                *ferr,
      double                *berr,
      double                *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("spbrfs");

    IndexType info;
    LAPACK_IMPL(dpbrfs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        Ab,
                        &ldAb,
                        Afb,
                        &ldAfb,
                        B,
                        &ldB,
                        X,
                        &ldX,
                        ferr,
                        berr,
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
pbrfs(char                        uplo,
      IndexType                   n,
      IndexType                   kd,
      IndexType                   nRhs,
      const std::complex<float >  *Ab,
      IndexType                   ldAb,
      const std::complex<float >  *Afb,
      IndexType                   ldAfb,
      const std::complex<float >  *B,
      IndexType                   ldB,
      std::complex<float >        *X,
      IndexType                   ldX,
      float                      *ferr,
      float                      *berr,
      std::complex<float >       *work,
      float                      *rWork)
{
    CXXLAPACK_DEBUG_OUT("cpbrfs");

    IndexType info;
    LAPACK_IMPL(cpbrfs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        reinterpret_cast<const float  *>(Ab),
                        &ldAb,
                        reinterpret_cast<const float  *>(Afb),
                        &ldAfb,
                        reinterpret_cast<const float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(X),
                        &ldX,
                        ferr,
                        berr,
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
pbrfs(char                        uplo,
      IndexType                   n,
      IndexType                   kd,
      IndexType                   nRhs,
      const std::complex<double>  *Ab,
      IndexType                   ldAb,
      const std::complex<double>  *Afb,
      IndexType                   ldAfb,
      const std::complex<double>  *B,
      IndexType                   ldB,
      std::complex<double>        *X,
      IndexType                   ldX,
      double                     *ferr,
      double                     *berr,
      std::complex<double>       *work,
      double                     *rWork)
{
    CXXLAPACK_DEBUG_OUT("zpbrfs");

    IndexType info;
    LAPACK_IMPL(zpbrfs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        reinterpret_cast<const double *>(Ab),
                        &ldAb,
                        reinterpret_cast<const double *>(Afb),
                        &ldAfb,
                        reinterpret_cast<const double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(X),
                        &ldX,
                        ferr,
                        berr,
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

#endif // CXXLAPACK_INTERFACE_PBRFS_TCC

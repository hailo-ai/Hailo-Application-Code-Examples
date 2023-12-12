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

#ifndef CXXLAPACK_INTERFACE_POSVX_TCC
#define CXXLAPACK_INTERFACE_POSVX_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
posvx(char                  fact,
      char                  uplo,
      IndexType             n,
      IndexType             nRhs,
      float                 *A,
      IndexType             ldA,
      float                 *Af,
      IndexType             ldAf,
      char                  &equed,
      float                 *s,
      float                 *B,
      IndexType             ldB,
      float                 *X,
      IndexType             ldX,
      float                 &rCond,
      float                 *ferr,
      float                 *berr,
      float                 *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("sposvx");

    IndexType info;
    LAPACK_IMPL(sposvx)(&fact,
                        &uplo,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        Af,
                        &ldAf,
                        &equed,
                        s,
                        B,
                        &ldB,
                        X,
                        &ldX,
                        &rCond,
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
posvx(char                  fact,
      char                  uplo,
      IndexType             n,
      IndexType             nRhs,
      double                *A,
      IndexType             ldA,
      double                *Af,
      IndexType             ldAf,
      char                  &equed,
      double                *s,
      double                *B,
      IndexType             ldB,
      double                *X,
      IndexType             ldX,
      double                &rCond,
      double                *ferr,
      double                *berr,
      double                *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("dposvx");

    IndexType info;
    LAPACK_IMPL(dposvx)(&fact,
                        &uplo,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        Af,
                        &ldAf,
                        &equed,
                        s,
                        B,
                        &ldB,
                        X,
                        &ldX,
                        &rCond,
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
posvx(char                  fact,
      char                  uplo,
      IndexType             n,
      IndexType             nRhs,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *Af,
      IndexType             ldAf,
      char                  &equed,
      float                 *s,
      std::complex<float >  *B,
      IndexType             ldB,
      std::complex<float >  *X,
      IndexType             ldX,
      float                 &rCond,
      float                 *ferr,
      float                 *berr,
      std::complex<float >  *work,
      float                 *rWork)
{
    CXXLAPACK_DEBUG_OUT("cposvx");

    IndexType info;
    LAPACK_IMPL(cposvx)(&fact,
                        &uplo,
                        &n,
                        &nRhs,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(Af),
                        &ldAf,
                        &equed,
                        s,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(X),
                        &ldX,
                        &rCond,
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
posvx(char                  fact,
      char                  uplo,
      IndexType             n,
      IndexType             nRhs,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *Af,
      IndexType             ldAf,
      char                  &equed,
      double                *s,
      std::complex<double>  *B,
      IndexType             ldB,
      std::complex<double>  *X,
      IndexType             ldX,
      double                &rCond,
      double                *ferr,
      double                *berr,
      std::complex<double>  *work,
      double                *rWork)
{
    CXXLAPACK_DEBUG_OUT("zposvx");

    IndexType info;
    LAPACK_IMPL(zposvx)(&fact,
                        &uplo,
                        &n,
                        &nRhs,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(Af),
                        &ldAf,
                        &equed,
                        s,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(X),
                        &ldX,
                        &rCond,
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

#endif // CXXLAPACK_INTERFACE_POSVX_TCC

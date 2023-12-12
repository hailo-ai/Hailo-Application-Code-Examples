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

#ifndef CXXLAPACK_INTERFACE_LALSD_TCC
#define CXXLAPACK_INTERFACE_LALSD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
lalsd(char                  uplo,
      IndexType             smlsiz,
      IndexType             n,
      IndexType             nRhs,
      float                 *d,
      float                 *e,
      float                 *B,
      IndexType             ldB,
      float                 rCond,
      IndexType             &rank,
      float                 *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("slalsd");

    IndexType info;
    LAPACK_IMPL(slalsd)(&uplo,
                        &smlsiz,
                        &n,
                        &nRhs,
                        d,
                        e,
                        B,
                        &ldB,
                        &rCond,
                        &rank,
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
lalsd(char                  uplo,
      IndexType             smlsiz,
      IndexType             n,
      IndexType             nRhs,
      double                *d,
      double                *e,
      double                *B,
      IndexType             ldB,
      double                rCond,
      IndexType             &rank,
      double                *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("dlalsd");

    IndexType info;
    LAPACK_IMPL(dlalsd)(&uplo,
                        &smlsiz,
                        &n,
                        &nRhs,
                        d,
                        e,
                        B,
                        &ldB,
                        &rCond,
                        &rank,
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
lalsd(char                  uplo,
      IndexType             smlsiz,
      IndexType             n,
      IndexType             nRhs,
      float                 *d,
      float                 *e,
      std::complex<float >  *B,
      IndexType             ldB,
      float                 rCond,
      IndexType             &rank,
      std::complex<float >  *work,
      float                 *rWork,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("clalsd");

    IndexType info;
    LAPACK_IMPL(clalsd)(&uplo,
                        &smlsiz,
                        &n,
                        &nRhs,
                        d,
                        e,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        &rCond,
                        &rank,
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
lalsd(char                  uplo,
      IndexType             smlsiz,
      IndexType             n,
      IndexType             nRhs,
      double                *d,
      double                *e,
      std::complex<double>  *B,
      IndexType             ldB,
      double                rCond,
      IndexType             &rank,
      std::complex<double>  *work,
      double                *rWork,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("zlalsd");

    IndexType info;
    LAPACK_IMPL(zlalsd)(&uplo,
                        &smlsiz,
                        &n,
                        &nRhs,
                        d,
                        e,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        &rCond,
                        &rank,
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

#endif // CXXLAPACK_INTERFACE_LALSD_TCC

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

#ifndef CXXLAPACK_INTERFACE_GGHRD_TCC
#define CXXLAPACK_INTERFACE_GGHRD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gghrd(char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      float                 *Q,
      IndexType             ldQ,
      float                 *Z,
      IndexType             ldZ)
{
    CXXLAPACK_DEBUG_OUT("sgghrd");

    IndexType info;
    LAPACK_IMPL(sgghrd)(&compq,
                        &compz
                        &n,
                        &ilo,
                        &ihi,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        Q,
                        &ldQ,
                        Z,
                        &ldZ,
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
gghrd(char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      double                *Q,
      IndexType             ldQ,
      double                *Z,
      IndexType             ldZ)
{
    CXXLAPACK_DEBUG_OUT("dgghrd");

    IndexType info;
    LAPACK_IMPL(dgghrd)(&compq,
                        &compz
                        &n,
                        &ilo,
                        &ihi,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        Q,
                        &ldQ,
                        Z,
                        &ldZ,
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
gghrd(char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      std::complex<float >  *Q,
      IndexType             ldQ,
      std::complex<float >  *Z,
      IndexType             ldZ)
{
    CXXLAPACK_DEBUG_OUT("cgghrd");

    IndexType info;
    LAPACK_IMPL(cgghrd)(&compq,
                        &compz
                        &n,
                        &ilo,
                        &ihi,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
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
gghrd(char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      std::complex<double>  *Q,
      IndexType             ldQ,
      std::complex<double>  *Z,
      IndexType             ldZ)
{
    CXXLAPACK_DEBUG_OUT("zgghrd");

    IndexType info;
    LAPACK_IMPL(zgghrd)(&compq,
                        &compz
                        &n,
                        &ilo,
                        &ihi,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
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

#endif // CXXLAPACK_INTERFACE_GGHRD_TCC

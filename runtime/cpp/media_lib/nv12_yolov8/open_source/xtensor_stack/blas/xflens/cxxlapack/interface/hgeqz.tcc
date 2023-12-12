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

#ifndef CXXLAPACK_INTERFACE_HGEQZ_TCC
#define CXXLAPACK_INTERFACE_HGEQZ_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
hgeqz(char                  job,
      char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      float                 *H,
      IndexType             ldH,
      float                 *T,
      IndexType             ldT,
      float                 *alphaar,
      float                 *alphaai,
      float                 *beta,
      float                 *Q,
      IndexType             ldQ,
      float                 *Z,
      IndexType             ldZ,
      float                 *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("shgeqz");

    IndexType info;
    LAPACK_IMPL(shgeqz)(&job,
                        &compq,
                        &compz,
                        &n,
                        &ilo,
                        &ihi,
                        H,
                        &ldH,
                        T,
                        &ldT,
                        alphaar,
                        alphaai,
                        beta,
                        Q,
                        &ldQ,
                        Z,
                        &ldZ,
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
hgeqz(char                  job,
      char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      double                *H,
      IndexType             ldH,
      double                *T,
      IndexType             ldT,
      double                *alphaar,
      double                *alphaai,
      double                *beta,
      double                *Q,
      IndexType             ldQ,
      double                *Z,
      IndexType             ldZ,
      double                *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("dhgeqz");

    IndexType info;
    LAPACK_IMPL(dhgeqz)(&job,
                        &compq,
                        &compz,
                        &n,
                        &ilo,
                        &ihi,
                        H,
                        &ldH,
                        T,
                        &ldT,
                        alphaar,
                        alphaai,
                        beta,
                        Q,
                        &ldQ,
                        Z,
                        &ldZ,
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
hgeqz(char                  job,
      char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      std::complex<float >  *H,
      IndexType             ldH,
      std::complex<float >  *T,
      IndexType             ldT,
      std::complex<float >  *alpha,
      std::complex<float >  *beta,
      std::complex<float >  *Q,
      IndexType             ldQ,
      std::complex<float >  *Z,
      IndexType             ldZ,
      std::complex<float >  *work,
      IndexType             lWork,
      float                 *rWork)
{
    CXXLAPACK_DEBUG_OUT("chgeqz");

    IndexType info;
    LAPACK_IMPL(chgeqz)(&job,
                        &compq,
                        &compz,
                        &n,
                        &ilo,
                        &ihi,
                        reinterpret_cast<float  *>(H),
                        &ldH,
                        reinterpret_cast<float  *>(T),
                        &ldT,
                        reinterpret_cast<float  *>(alpha),
                        reinterpret_cast<float  *>(beta),
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
                        reinterpret_cast<float  *>(work),
                        &lWork,
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
hgeqz(char                  job,
      char                  compq,
      char                  compz,
      IndexType             n,
      IndexType             ilo,
      IndexType             ihi,
      std::complex<double>  *H,
      IndexType             ldH,
      std::complex<double>  *T,
      IndexType             ldT,
      std::complex<double>  *alpha,
      std::complex<double>  *beta,
      std::complex<double>  *Q,
      IndexType             ldQ,
      std::complex<double>  *Z,
      IndexType             ldZ,
      std::complex<double>  *work,
      IndexType             lWork,
      double                *rWork)
{
    CXXLAPACK_DEBUG_OUT("zhgeqz");

    IndexType info;
    LAPACK_IMPL(zhgeqz)(&job,
                        &compq,
                        &compz,
                        &n,
                        &ilo,
                        &ihi,
                        reinterpret_cast<double *>(H),
                        &ldH,
                        reinterpret_cast<double *>(T),
                        &ldT,
                        reinterpret_cast<double *>(alpha),
                        reinterpret_cast<double *>(beta),
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
                        reinterpret_cast<double *>(work),
                        &lWork,
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

#endif // CXXLAPACK_INTERFACE_HGEQZ_TCC

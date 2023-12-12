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

#ifndef CXXLAPACK_INTERFACE_TGEXC_TCC
#define CXXLAPACK_INTERFACE_TGEXC_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
tgexc(bool                  wantq,
      bool                  wantz,
      IndexType             n,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      float                 *Q,
      IndexType             ldQ,
      float                 *Z,
      IndexType             ldZ,
      IndexType             ifst,
      IndexType             ilst,
      float                 *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("stgexc");

    IndexType info;
    IndexType wantq_ = wantq;
    IndexType wantz_ = wantz;
    LAPACK_IMPL(stgexc)(&wantq_,
                        &wantz_,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        Q,
                        &ldQ,
                        Z,
                        &ldZ,
                        &ifst,
                        &ilst,
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
tgexc(bool                  wantq,
      bool                  wantz,
      IndexType             n,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      double                *Q,
      IndexType             ldQ,
      double                *Z,
      IndexType             ldZ,
      IndexType             ifst,
      IndexType             ilst,
      double                *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("dtgexc");

    IndexType info;
    IndexType wantq_ = wantq;
    IndexType wantz_ = wantz;
    LAPACK_IMPL(dtgexc)(&wantq_,
                        &wantz_,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        Q,
                        &ldQ,
                        Z,
                        &ldZ,
                        &ifst,
                        &ilst,
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
tgexc(bool                  wantq,
      bool                  wantz,
      IndexType             n,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      std::complex<float >  *Q,
      IndexType             ldQ,
      std::complex<float >  *Z,
      IndexType             ldZ,
      IndexType             ifst,
      IndexType             ilst)
{
    CXXLAPACK_DEBUG_OUT("ctgexc");

    IndexType info;
    IndexType wantq_ = wantq;
    IndexType wantz_ = wantz;
    LAPACK_IMPL(ctgexc)(&wantq_,
                        &wantz_,
                        &n,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
                        &ifst,
                        &ilst,
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
tgexc(bool                  wantq,
      bool                  wantz,
      IndexType             n,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      std::complex<double>  *Q,
      IndexType             ldQ,
      std::complex<double>  *Z,
      IndexType             ldZ,
      IndexType             ifst,
      IndexType             ilst)
{
    CXXLAPACK_DEBUG_OUT("ztgexc");

    IndexType info;
    IndexType wantq_ = wantq;
    IndexType wantz_ = wantz;
    LAPACK_IMPL(ztgexc)(&wantq_,
                        &wantz_,
                        &n,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
                        &ifst,
                        &ilst,
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

#endif // CXXLAPACK_INTERFACE_TGEXC_TCC

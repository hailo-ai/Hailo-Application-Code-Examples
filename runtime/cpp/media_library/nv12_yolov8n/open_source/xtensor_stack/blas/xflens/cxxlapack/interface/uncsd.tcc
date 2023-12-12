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

#ifndef CXXLAPACK_INTERFACE_UNCSD_TCC
#define CXXLAPACK_INTERFACE_UNCSD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {
template <typename IndexType>
IndexType
uncsd(char                         jobu1,
      char                         jobu2,
      char                         jobv1t,
      char                         jobv2t,
      char                         trans,
      char                         signs,
      IndexType                    m,
      IndexType                    p,
      IndexType                    q,
      const std::complex<float >   *X11,
      IndexType                    ldX11,
      const std::complex<float >   *X12,
      IndexType                    ldX12,
      const std::complex<float >   *X21,
      IndexType                    ldX21,
      const std::complex<float >   *X22,
      IndexType                    ldX22,
      float                        *theta,
      std::complex<float >         *U1,
      IndexType                    ldU1,
      std::complex<float >         *U2,
      IndexType                    ldU2,
      std::complex<float >         *V1t,
      IndexType                    ldV1t,
      std::complex<float >         *V2t,
      IndexType                    ldV2t,
      std::complex<float >         *work,
      IndexType                    lWork,
      float                        *rWork,
      IndexType                    lrWork,
      IndexType                    *iWork)
{
    CXXLAPACK_DEBUG_OUT("cuncsd");

    IndexType info;
    LAPACK_IMPL(cuncsd)(&jobu1,
                        &jobu2,
                        &jobv1t,
                        &jobv2t,
                        &trans,
                        &signs,
                        &m,
                        &p,
                        &q,
                        reinterpret_cast<const float  *>(X11),
                        &ldX11,
                        reinterpret_cast<const float  *>(X12),
                        &ldX12,
                        reinterpret_cast<const float  *>(X21),
                        &ldX21,
                        reinterpret_cast<const float  *>(X22),
                        &ldX22,
                        theta,
                        reinterpret_cast<float  *>(U1),
                        &ldU1,
                        reinterpret_cast<float  *>(U2),
                        &ldU2,
                        reinterpret_cast<float  *>(V1t),
                        &ldV1t,
                        reinterpret_cast<float  *>(V2t),
                        &ldV2t,
                        reinterpret_cast<float  *>(work),
                        &lWork,
                        rWork,
                        &lrWork,
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
uncsd(char                         jobu1,
      char                         jobu2,
      char                         jobv1t,
      char                         jobv2t,
      char                         trans,
      char                         signs,
      IndexType                    m,
      IndexType                    p,
      IndexType                    q,
      const std::complex<double>   *X11,
      IndexType                    ldX11,
      const std::complex<double>   *X12,
      IndexType                    ldX12,
      const std::complex<double>   *X21,
      IndexType                    ldX21,
      const std::complex<double>   *X22,
      IndexType                    ldX22,
      double                       *theta,
      std::complex<double>         *U1,
      IndexType                    ldU1,
      std::complex<double>         *U2,
      IndexType                    ldU2,
      std::complex<double>         *V1t,
      IndexType                    ldV1t,
      std::complex<double>         *V2t,
      IndexType                    ldV2t,
      std::complex<double>         *work,
      IndexType                    lWork,
      double                       *rWork,
      IndexType                    lrWork,
      IndexType                    *iWork)
{
    CXXLAPACK_DEBUG_OUT("zuncsd");

    IndexType info;
    LAPACK_IMPL(zuncsd)(&jobu1,
                        &jobu2,
                        &jobv1t,
                        &jobv2t,
                        &trans,
                        &signs,
                        &m,
                        &p,
                        &q,
                        reinterpret_cast<const double *>(X11),
                        &ldX11,
                        reinterpret_cast<const double *>(X12),
                        &ldX12,
                        reinterpret_cast<const double *>(X21),
                        &ldX21,
                        reinterpret_cast<const double *>(X22),
                        &ldX22,
                        theta,
                        reinterpret_cast<double *>(U1),
                        &ldU1,
                        reinterpret_cast<double *>(U2),
                        &ldU2,
                        reinterpret_cast<double *>(V1t),
                        &ldV1t,
                        reinterpret_cast<double *>(V2t),
                        &ldV2t,
                        reinterpret_cast<double *>(work),
                        &lWork,
                        rWork,
                        &lrWork,
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

#endif // CXXLAPACK_INTERFACE_UNCSD_TCC

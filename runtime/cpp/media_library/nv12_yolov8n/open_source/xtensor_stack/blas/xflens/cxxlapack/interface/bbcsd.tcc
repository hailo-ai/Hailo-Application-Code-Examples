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

#ifndef CXXLAPACK_INTERFACE_BBCSD_TCC
#define CXXLAPACK_INTERFACE_BBCSD_TCC 1

#include <complex>
#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
bbcsd(char                  jobu1,
      char                  jobu2,
      char                  jobv1t,
      char                  jobv2t,
      char                  trans,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      float                 *theta,
      float                 *phi,
      float                 *U1,
      IndexType             ldU1,
      float                 *U2,
      IndexType             ldU2,
      float                 *V1t,
      IndexType             ldV1t,
      float                 *V2t,
      IndexType             ldV2t,
      const float           *b11d,
      const float           *b11e,
      const float           *b12d,
      const float           *b12e,
      float                 *work,
      IndexType             lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sbbcsd");
    LAPACK_IMPL(sbbcsd)(&jobu1,
                        &jobu2,
                        &jobv1t,
                        &jobv2t,
                        &trans,
                        &m,
                        &p,
                        &q,
                        theta,
                        phi,
                        U1,
                        &ldU1,
                        U2,
                        &ldU2,
                        V1t,
                        &ldV1t,
                        V2t,
                        &ldV2t,
                        b11d,
                        b11e,
                        b12d,
                        b12e,
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
bbcsd(char                  jobu1,
      char                  jobu2,
      char                  jobv1t,
      char                  jobv2t,
      char                  trans,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      double                *theta,
      double                *phi,
      double                *U1,
      IndexType             ldU1,
      double                *U2,
      IndexType             ldU2,
      double                *V1t,
      IndexType             ldV1t,
      double                *V2t,
      IndexType             ldV2t,
      const double          *b11d,
      const double          *b11e,
      const double          *b12d,
      const double          *b12e,
      double                *work,
      IndexType             lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dbbcsd");
    LAPACK_IMPL(dbbcsd)(&jobu1,
                        &jobu2,
                        &jobv1t,
                        &jobv2t,
                        &trans,
                        &m,
                        &p,
                        &q,
                        theta,
                        phi,
                        U1,
                        &ldU1,
                        U2,
                        &ldU2,
                        V1t,
                        &ldV1t,
                        V2t,
                        &ldV2t,
                        b11d,
                        b11e,
                        b12d,
                        b12e,
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
bbcsd(char                  jobu1,
      char                  jobu2,
      char                  jobv1t,
      char                  jobv2t,
      char                  trans,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      float                 *theta,
      float                 *phi,
      std::complex<float>   *U1,
      IndexType             ldU1,
      std::complex<float>   *U2,
      IndexType             ldU2,
      std::complex<float>   *V1t,
      IndexType             ldV1t,
      std::complex<float>   *V2t,
      IndexType             ldV2t,
      const float           *b11d,
      const float           *b11e,
      const float           *b12d,
      const float           *b12e,
      float                 *work,
      IndexType             lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cbbcsd");
    LAPACK_IMPL(cbbcsd)(&jobu1,
                        &jobu2,
                        &jobv1t,
                        &jobv2t,
                        &trans,
                        &m,
                        &p,
                        &q,
                        theta,
                        phi,
                        reinterpret_cast<float  *>(U1),
                        &ldU1,
                        reinterpret_cast<float  *>(U2),
                        &ldU2,
                        reinterpret_cast<float  *>(V1t),
                        &ldV1t,
                        reinterpret_cast<float  *>(V2t),
                        &ldV2t,
                        b11d,
                        b11e,
                        b12d,
                        b12e,
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
bbcsd(char                  jobu1,
      char                  jobu2,
      char                  jobv1t,
      char                  jobv2t,
      char                  trans,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      double                *theta,
      double                *phi,
      std::complex<double>  *U1,
      IndexType             ldU1,
      std::complex<double>  *U2,
      IndexType             ldU2,
      std::complex<double>  *V1t,
      IndexType             ldV1t,
      std::complex<double>  *V2t,
      IndexType             ldV2t,
      const double          *b11d,
      const double          *b11e,
      const double          *b12d,
      const double          *b12e,
      double                *work,
      IndexType             lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zbbcsd");
    LAPACK_IMPL(zbbcsd)(&jobu1,
                        &jobu2,
                        &jobv1t,
                        &jobv2t,
                        &trans,
                        &m,
                        &p,
                        &q,
                        theta,
                        phi,
                        reinterpret_cast<double *>(U1),
                        &ldU1,
                        reinterpret_cast<double *>(U2),
                        &ldU2,
                        reinterpret_cast<double *>(V1t),
                        &ldV1t,
                        reinterpret_cast<double *>(V2t),
                        &ldV2t,
                        b11d,
                        b11e,
                        b12d,
                        b12e,
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

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_BBCSD_TCC

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

#ifndef CXXLAPACK_INTERFACE_TGEVC_TCC
#define CXXLAPACK_INTERFACE_TGEVC_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
tgevc(char                  side,
      char                  howmny,
      bool                  select,
      IndexType             n,
      const float           *S,
      IndexType             ldS,
      const float           *P,
      IndexType             ldP,
      float                 *VL,
      IndexType             ldVL,
      float                 *VR,
      IndexType             ldVR,
      IndexType             mm,
      IndexType             &m,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("stgevc");

    IndexType info;
    IndexType select_ = select;
    LAPACK_IMPL(stgevc)(&side,
                        &howmny,
                        &select_,
                        &n,
                        S,
                        &ldS,
                        P,
                        &ldP,
                        VL,
                        &ldVL,
                        VR,
                        &ldVR,
                        &mm,
                        &m,
                        work,
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
tgevc(char                  side,
      char                  howmny,
      bool                  select,
      IndexType             n,
      const double          *S,
      IndexType             ldS,
      const double          *P,
      IndexType             ldP,
      double                *VL,
      IndexType             ldVL,
      double                *VR,
      IndexType             ldVR,
      IndexType             mm,
      IndexType             &m,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dtgevc");

    IndexType info;
    IndexType select_ = select;
    LAPACK_IMPL(dtgevc)(&side,
                        &howmny,
                        &select_,
                        &n,
                        S,
                        &ldS,
                        P,
                        &ldP,
                        VL,
                        &ldVL,
                        VR,
                        &ldVR,
                        &mm,
                        &m,
                        work,
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
tgevc(char                        side,
      char                        howmny,
      bool                        select,
      IndexType                   n,
      const std::complex<float >  *S,
      IndexType                   ldS,
      const std::complex<float >  *P,
      IndexType                   ldP,
      std::complex<float >        *VL,
      IndexType                   ldVL,
      std::complex<float >        *VR,
      IndexType                   ldVR,
      IndexType                   mm,
      IndexType                   &m,
      std::complex<float >        *work,
      double                      *rWork)
{
    CXXLAPACK_DEBUG_OUT("ctgevc");

    IndexType info;
    IndexType select_ = select;
    LAPACK_IMPL(ctgevc)(&side,
                        &howmny,
                        &select_,
                        &n,
                        reinterpret_cast<const float  *>(S),
                        &ldS,
                        reinterpret_cast<const float  *>(P),
                        &ldP,
                        reinterpret_cast<float  *>(VL),
                        &ldVL,
                        reinterpret_cast<float  *>(VR),
                        &ldVR,
                        &mm,
                        &m,
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
tgevc(char                        side,
      char                        howmny,
      bool                        select,
      IndexType                   n,
      const std::complex<double>  *S,
      IndexType                   ldS,
      const std::complex<double>  *P,
      IndexType                   ldP,
      std::complex<double>        *VL,
      IndexType                   ldVL,
      std::complex<double>        *VR,
      IndexType                   ldVR,
      IndexType                   mm,
      IndexType                   &m,
      std::complex<double>        *work,
      double                      *rWork)
{
    CXXLAPACK_DEBUG_OUT("ztgevc");

    IndexType info;
    IndexType select_ = select;
    LAPACK_IMPL(ztgevc)(&side,
                        &howmny,
                        &select_,
                        &n,
                        reinterpret_cast<const double *>(S),
                        &ldS,
                        reinterpret_cast<const double *>(P),
                        &ldP,
                        reinterpret_cast<double *>(VL),
                        &ldVL,
                        reinterpret_cast<double *>(VR),
                        &ldVR,
                        &mm,
                        &m,
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

#endif // CXXLAPACK_INTERFACE_TGEVC_TCC

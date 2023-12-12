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

#ifndef CXXLAPACK_INTERFACE_BDSQR_TCC
#define CXXLAPACK_INTERFACE_BDSQR_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
bdsqr(char                  upLo,
      IndexType             n,
      IndexType             ncvt,
      IndexType             nru,
      IndexType             ncc,
      float                 *d,
      float                 *e,
      float                 *VT,
      IndexType             ldVT,
      float                 *U,
      IndexType             ldU,
      float                 *C,
      IndexType             ldC,
      float                 *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sbdsqr");
    LAPACK_IMPL(sbdsqr)(&upLo,
                        &n,
                        &ncvt,
                        &nru,
                        &ncc,
                        d,
                        e,
                        VT,
                        &ldVT,
                        U,
                        &ldU,
                        C,
                        &ldC,
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
bdsqr(char                  upLo,
      IndexType             n,
      IndexType             ncvt,
      IndexType             nru,
      IndexType             ncc,
      double                *d,
      double                *e,
      double                *VT,
      IndexType             ldVT,
      double                *U,
      IndexType             ldU,
      double                *C,
      IndexType             ldC,
      double                *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dbdsqr");
    LAPACK_IMPL(dbdsqr)(&upLo,
                        &n,
                        &ncvt,
                        &nru,
                        &ncc,
                        d,
                        e,
                        VT,
                        &ldVT,
                        U,
                        &ldU,
                        C,
                        &ldC,
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
bdsqr(char                  upLo,
      IndexType             n,
      IndexType             ncvt,
      IndexType             nru,
      IndexType             ncc,
      float                 *d,
      float                 *e,
      std::complex<float >  *VT,
      IndexType             ldVT,
      std::complex<float >  *U,
      IndexType             ldU,
      std::complex<float >  *C,
      IndexType             ldC,
      float                 *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cbdsqr");
    LAPACK_IMPL(cbdsqr)(&upLo,
                        &n,
                        &ncvt,
                        &nru,
                        &ncc,
                        d,
                        e,
                        reinterpret_cast<float  *>(VT),
                        &ldVT,
                        reinterpret_cast<float  *>(U),
                        &ldU,
                        reinterpret_cast<float  *>(C),
                        &ldC,
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
bdsqr(char                  upLo,
      IndexType             n,
      IndexType             ncvt,
      IndexType             nru,
      IndexType             ncc,
      double                *d,
      double                *e,
      std::complex<double>  *VT,
      IndexType             ldVT,
      std::complex<double>  *U,
      IndexType             ldU,
      std::complex<double>  *C,
      IndexType             ldC,
      double                *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zbdsqr");
    LAPACK_IMPL(zbdsqr)(&upLo,
                        &n,
                        &ncvt,
                        &nru,
                        &ncc,
                        d,
                        e,
                        reinterpret_cast<double *>(VT),
                        &ldVT,
                        reinterpret_cast<double *>(U),
                        &ldU,
                        reinterpret_cast<double *>(C),
                        &ldC,
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

#endif // CXXLAPACK_INTERFACE_BDSQR_TCC

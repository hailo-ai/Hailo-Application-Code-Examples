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

#ifndef CXXLAPACK_INTERFACE_LARRF_TCC
#define CXXLAPACK_INTERFACE_LARRF_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
larrf(char                  range,
      const float           *d,
      const float           *l,
      const float           *ld,
      IndexType             clstrt,
      IndexType             clend,
      const float           *w,
      float                 *wgap,
      const float           *werr,
      float                 spdiam,
      float                 clgapl,
      float                 clgapr,
      float                 pivmin,
      float                 &sigma,
      float                 *dplus,
      float                 *lplus,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("slarrf");

    IndexType info;
    LAPACK_IMPL(slarrf)(&range,
                        d,
                        l,
                        ld,
                        &clstrt,
                        &clend,
                        w,
                        wgap,
                        werr,
                        &spdiam,
                        &clgapl,
                        &clgapr,
                        &pivmin,
                        &sigma,
                        dplus,
                        lplus,
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
larrf(char                  range,
      const double          *d,
      const double          *l,
      const double          *ld,
      IndexType             clstrt,
      IndexType             clend,
      const double          *w,
      double                *wgap,
      const double          *werr,
      double                spdiam,
      double                clgapl,
      double                clgapr,
      double                pivmin,
      double                &sigma,
      double                *dplus,
      double                *lplus,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dlarrf");

    IndexType info;
    LAPACK_IMPL(dlarrf)(&range,
                        d,
                        l,
                        ld,
                        &clstrt,
                        &clend,
                        w,
                        wgap,
                        werr,
                        &spdiam,
                        &clgapl,
                        &clgapr,
                        &pivmin,
                        &sigma,
                        dplus,
                        lplus,
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

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARRF_TCC

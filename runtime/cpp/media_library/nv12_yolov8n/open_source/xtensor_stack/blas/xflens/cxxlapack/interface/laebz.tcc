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

#ifndef CXXLAPACK_INTERFACE_LAEBZ_TCC
#define CXXLAPACK_INTERFACE_LAEBZ_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
laebz(IndexType             ijob,
      IndexType             nitmax,
      IndexType             n,
      IndexType             mmax,
      IndexType             minp,
      IndexType             nbmin,
      float                 abstol,
      float                  reltol,
      float                 pivmin,
      const float           *d,
      const float           *e,
      const float           *e2,
      IndexType             *nval,
      float                 *Ab,
      float                 *c,
      IndexType             &mout,
      IndexType             *NAb,
      float                 *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("slaebz");

    IndexType info;
    LAPACK_IMPL(slaebz)(&ijob,
                        &nitmax,
                        &n,
                        &mmax,
                        &minp,
                        &nbmin,
                        &abstol,
                        &reltol,
                        &pivmin,
                        d,
                        e,
                        e2,
                        nval,
                        Ab,
                        c,
                        &mout,
                        NAb,
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
laebz(IndexType             ijob,
      IndexType             nitmax,
      IndexType             n,
      IndexType             mmax,
      IndexType             minp,
      IndexType             nbmin,
      double                abstol,
      double                reltol,
      double                pivmin,
      const double          *d,
      const double          *e,
      const double          *e2,
      IndexType             *nval,
      double                *Ab,
      double                *c,
      IndexType             &mout,
      IndexType             *NAb,
      double                *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("dlaebz");

    IndexType info;
    LAPACK_IMPL(dlaebz)(&ijob,
                        &nitmax,
                        &n,
                        &mmax,
                        &minp,
                        &nbmin,
                        &abstol,
                        &reltol,
                        &pivmin,
                        d,
                        e,
                        e2,
                        nval,
                        Ab,
                        c,
                        &mout,
                        NAb,
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

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAEBZ_TCC

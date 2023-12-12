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

#ifndef CXXLAPACK_INTERFACE_LASD3_TCC
#define CXXLAPACK_INTERFACE_LASD3_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
lasd3(IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             k,
      float                 *d,
      float                 *D,
      IndexType             ldD,
      const float           dsigma,
      float                 *U,
      IndexType             ldU,
      float                 *U2,
      IndexType             ldU2,
      float                 *Vt,
      IndexType             ldVt,
      float                 *Vt2,
      IndexType             ldVt2,
      const IndexType       *idxc,
      const IndexType       *ctot,
      const float           *z)
{
    CXXLAPACK_DEBUG_OUT("slasd3");

    IndexType info;
    LAPACK_IMPL(slasd3)(&nl,
                        &nr,
                        &sqre,
                        &k,
                        d,
                        D,
                        &ldD,
                        &dsigma,
                        U,
                        &ldU,
                        U2,
                        &ldU2,
                        Vt,
                        &ldVt,
                        Vt2,
                        &ldVt2,
                        idxc,
                        ctot,
                        z,
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
lasd3(IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             k,
      double                *d,
      double                *D,
      IndexType             ldD,
      const double          dsigma,
      double                *U,
      IndexType             ldU,
      double                *U2,
      IndexType             ldU2,
      double                *Vt,
      IndexType             ldVt,
      double                *Vt2,
      IndexType             ldVt2,
      const IndexType       *idxc,
      const IndexType       *ctot,
      const double          *z)
{
    CXXLAPACK_DEBUG_OUT("dlasd3");

    IndexType info;
    LAPACK_IMPL(dlasd3)(&nl,
                        &nr,
                        &sqre,
                        &k,
                        d,
                        D,
                        &ldD,
                        &dsigma,
                        U,
                        &ldU,
                        U2,
                        &ldU2,
                        Vt,
                        &ldVt,
                        Vt2,
                        &ldVt2,
                        idxc,
                        ctot,
                        z,
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

#endif // CXXLAPACK_INTERFACE_LASD3_TCC

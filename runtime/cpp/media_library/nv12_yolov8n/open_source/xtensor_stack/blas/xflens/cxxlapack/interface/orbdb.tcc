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

#ifndef CXXLAPACK_INTERFACE_ORBDB_TCC
#define CXXLAPACK_INTERFACE_ORBDB_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
orbdb(char                  trans,
      char                  signs,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      float                 *X11,
      IndexType             ldX11,
      float                 *X12,
      IndexType             ldX12,
      float                 *X21,
      IndexType             ldX21,
      float                 *X22,
      IndexType             ldX22,
      float                 *theta,
      float                 *phi,
      float                 *taup1,
      float                 *taup2,
      float                 *tauq1,
      float                 *tauq2,
      float                 *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("sorbdb");

    IndexType info;
    LAPACK_IMPL(sorbdb)(&trans,
                        &signs,
                        &m,
                        &p,
                        &q,
                        X11,
                        &ldX11,
                        X12,
                        &ldX12,
                        X21,
                        &ldX21,
                        X22,
                        &ldX22,
                        theta,
                        phi,
                        taup1,
                        taup2,
                        tauq1,
                        tauq2,
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
orbdb(char                  trans,
      char                  signs,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      double                *X11,
      IndexType             ldX11,
      double                *X12,
      IndexType             ldX12,
      double                *X21,
      IndexType             ldX21,
      double                *X22,
      IndexType             ldX22,
      double                *theta,
      double                *phi,
      double                *taup1,
      double                *taup2,
      double                *tauq1,
      double                *tauq2,
      double                *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("dorbdb");

    IndexType info;
    LAPACK_IMPL(dorbdb)(&trans,
                        &signs,
                        &m,
                        &p,
                        &q,
                        X11,
                        &ldX11,
                        X12,
                        &ldX12,
                        X21,
                        &ldX21,
                        X22,
                        &ldX22,
                        theta,
                        phi,
                        taup1,
                        taup2,
                        tauq1,
                        tauq2,
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

#endif // CXXLAPACK_INTERFACE_ORBDB_TCC

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

#ifndef CXXLAPACK_INTERFACE_UNBDB_TCC
#define CXXLAPACK_INTERFACE_UNBDB_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
unbdb(char                  trans,
      char                  signs,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      std::complex<float >  *X11,
      IndexType             ldX11,
      std::complex<float >  *X12,
      IndexType             ldX12,
      std::complex<float >  *X21,
      IndexType             ldX21,
      std::complex<float >  *X22,
      IndexType             ldX22,
      float                 *theta,
      float                 *phi,
      std::complex<float >  *taup1,
      std::complex<float >  *taup2,
      std::complex<float >  *tauq1,
      std::complex<float >  *tauq2,
      std::complex<float >  *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("cunbdb");

    IndexType info;
    LAPACK_IMPL(cunbdb)(&trans,
                        &signs
                        &m,
                        &p,
                        &q,
                        reinterpret_cast<float  *>(X11),
                        &ldX11,
                        reinterpret_cast<float  *>(X12),
                        &ldX12,
                        reinterpret_cast<float  *>(X21),
                        &ldX21,
                        reinterpret_cast<float  *>(X22),
                        &ldX22,
                        theta,
                        phi,
                        reinterpret_cast<float  *>(taup1),
                        reinterpret_cast<float  *>(taup2),
                        reinterpret_cast<float  *>(tauq1),
                        reinterpret_cast<float  *>(tauq2),
                        reinterpret_cast<float  *>(work),
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
unbdb(char                  trans,
      char                  signs,
      IndexType             m,
      IndexType             p,
      IndexType             q,
      std::complex<double>  *X11,
      IndexType             ldX11,
      std::complex<double>  *X12,
      IndexType             ldX12,
      std::complex<double>  *X21,
      IndexType             ldX21,
      std::complex<double>  *X22,
      IndexType             ldX22,
      double                *theta,
      double                *phi,
      std::complex<double>  *taup1,
      std::complex<double>  *taup2,
      std::complex<double>  *tauq1,
      std::complex<double>  *tauq2,
      std::complex<double>  *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("zunbdb");

    IndexType info;
    LAPACK_IMPL(zunbdb)(&trans,
                        &signs
                        &m,
                        &p,
                        &q,
                        reinterpret_cast<double *>(X11),
                        &ldX11,
                        reinterpret_cast<double *>(X12),
                        &ldX12,
                        reinterpret_cast<double *>(X21),
                        &ldX21,
                        reinterpret_cast<double *>(X22),
                        &ldX22,
                        theta,
                        phi,
                        reinterpret_cast<double *>(taup1),
                        reinterpret_cast<double *>(taup2),
                        reinterpret_cast<double *>(tauq1),
                        reinterpret_cast<double *>(tauq2),
                        reinterpret_cast<double *>(work),
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

#endif // CXXLAPACK_INTERFACE_UNBDB_TCC

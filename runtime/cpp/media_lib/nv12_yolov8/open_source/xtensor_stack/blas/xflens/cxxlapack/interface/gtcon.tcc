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

#ifndef CXXLAPACK_INTERFACE_GTCON_TCC
#define CXXLAPACK_INTERFACE_GTCON_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gtcon(char                  norm,
      IndexType             n,
      const float           *dl,
      const float           *d,
      const float           *du,
      const float           *du2,
      const IndexType       *iPiv,
      float                 anorm,
      float                 &rCond,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("sgtcon");

    IndexType info;
    LAPACK_IMPL(sgtcon)(&norm,
                        &n,
                        dl,
                        d,
                        du,
                        du2,
                        iPiv,
                        &anorm,
                        &rCond,
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
gtcon(char                  norm,
      IndexType             n,
      const double          *dl,
      const double          *d,
      const double          *du,
      const double          *du2,
      const IndexType       *iPiv,
      double                anorm,
      double                &rCond,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dgtcon");

    IndexType info;
    LAPACK_IMPL(dgtcon)(&norm,
                        &n,
                        dl,
                        d,
                        du,
                        du2,
                        iPiv,
                        &anorm,
                        &rCond,
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
gtcon(char                        norm,
      IndexType                   n,
      const std::complex<float >  *dl,
      const std::complex<float >  *d,
      const std::complex<float >  *du,
      const std::complex<float >  *du2,
      const IndexType             *iPiv,
      float                       anorm,
      float                       &rCond,
      std::complex<float >       *work)
{
    CXXLAPACK_DEBUG_OUT("cgtcon");

    IndexType info;
    LAPACK_IMPL(cgtcon)(&norm,
                        &n,
                        reinterpret_cast<const float  *>(dl),
                        reinterpret_cast<const float  *>(d),
                        reinterpret_cast<const float  *>(du),
                        reinterpret_cast<const float  *>(du2),
                        iPiv,
                        &anorm,
                        &rCond,
                        reinterpret_cast<float  *>(work),
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
gtcon(char                        norm,
      IndexType                   n,
      const std::complex<double>  *dl,
      const std::complex<double>  *d,
      const std::complex<double>  *du,
      const std::complex<double>  *du2,
      const IndexType             *iPiv,
      double                      anorm,
      double                      &rCond,
      std::complex<double>       *work)
{
    CXXLAPACK_DEBUG_OUT("zgtcon");

    IndexType info;
    LAPACK_IMPL(zgtcon)(&norm,
                        &n,
                        reinterpret_cast<const double *>(dl),
                        reinterpret_cast<const double *>(d),
                        reinterpret_cast<const double *>(du),
                        reinterpret_cast<const double *>(du2),
                        iPiv,
                        &anorm,
                        &rCond,
                        reinterpret_cast<double *>(work),
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

#endif // CXXLAPACK_INTERFACE_GTCON_TCC

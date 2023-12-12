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

#ifndef CXXLAPACK_INTERFACE_GTTRF_TCC
#define CXXLAPACK_INTERFACE_GTTRF_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gttrf(IndexType             n,
      float                 *dl,
      float                 *d,
      float                 *du,
      float                 *du2,
      IndexType             *iPiv)
{
    CXXLAPACK_DEBUG_OUT("sgttrf");

    IndexType info;
    LAPACK_IMPL(sgttrf)(&n,
                        dl,
                        d,
                        du,
                        du2,
                        iPiv,
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
gttrf(IndexType             n,
      double                *dl,
      double                *d,
      double                *du,
      double                *du2,
      IndexType             *iPiv)
{
    CXXLAPACK_DEBUG_OUT("dgttrf");

    IndexType info;
    LAPACK_IMPL(dgttrf)(&n,
                        dl,
                        d,
                        du,
                        du2,
                        iPiv,
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
gttrf(IndexType             n,
      std::complex<float >  *dl,
      std::complex<float >  *d,
      std::complex<float >  *du,
      std::complex<float >  *du2,
      IndexType             *iPiv)
{
    CXXLAPACK_DEBUG_OUT("cgttrf");

    IndexType info;
    LAPACK_IMPL(cgttrf)(&n,
                        reinterpret_cast<float  *>(dl),
                        reinterpret_cast<float  *>(d),
                        reinterpret_cast<float  *>(du),
                        reinterpret_cast<float  *>(du2),
                        iPiv,
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
gttrf(IndexType             n,
      std::complex<double>  *dl,
      std::complex<double>  *d,
      std::complex<double>  *du,
      std::complex<double>  *du2,
      IndexType             *iPiv)
{
    CXXLAPACK_DEBUG_OUT("zgttrf");

    IndexType info;
    LAPACK_IMPL(zgttrf)(&n,
                        reinterpret_cast<double *>(dl),
                        reinterpret_cast<double *>(d),
                        reinterpret_cast<double *>(du),
                        reinterpret_cast<double *>(du2),
                        iPiv,
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

#endif // CXXLAPACK_INTERFACE_GTTRF_TCC

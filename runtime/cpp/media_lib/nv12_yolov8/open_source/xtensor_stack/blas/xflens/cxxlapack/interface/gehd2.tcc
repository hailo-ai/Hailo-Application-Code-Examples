/*
 *   Copyright (c) 2012, Michael Lehn
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

#ifndef CXXLAPACK_INTERFACE_GEHD2_TCC
#define CXXLAPACK_INTERFACE_GEHD2_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gehd2(IndexType    n,
      IndexType    iLo,
      IndexType    iHi,
      float        *A,
      IndexType    ldA,
      float        *tau,
      float        *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgehd2");
    LAPACK_IMPL(sgehd2)(&n,
                        &iLo,
                        &iHi,
                        A,
                        &ldA,
                        tau,
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
gehd2(IndexType    n,
      IndexType    iLo,
      IndexType    iHi,
      double       *A,
      IndexType    ldA,
      double       *tau,
      double       *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgehd2");
    LAPACK_IMPL(dgehd2)(&n,
                        &iLo,
                        &iHi,
                        A,
                        &ldA,
                        tau,
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
gehd2(IndexType             n,
      IndexType             iLo,
      IndexType             iHi,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *tau,
      std::complex<float >  *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgehd2");
    LAPACK_IMPL(cgehd2)(&n,
                        &iLo,
                        &iHi,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(tau),
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
gehd2(IndexType             n,
      IndexType             iLo,
      IndexType             iHi,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *tau,
      std::complex<double>  *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgehd2");
    LAPACK_IMPL(zgehd2)(&n,
                        &iLo,
                        &iHi,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(tau),
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

#endif // CXXLAPACK_INTERFACE_GEHD2_TCC

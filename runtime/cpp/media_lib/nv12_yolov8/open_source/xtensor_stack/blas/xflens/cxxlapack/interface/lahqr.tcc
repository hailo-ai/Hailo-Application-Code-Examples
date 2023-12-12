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

#ifndef CXXLAPACK_INTERFACE_LAHQR_TCC
#define CXXLAPACK_INTERFACE_LAHQR_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
lahqr(bool          wantT,
      bool          wantZ,
      IndexType     n,
      IndexType     iLo,
      IndexType     iHi,
      float         *H,
      IndexType     ldH,
      float         *wr,
      float         *wi,
      IndexType     iLoZ,
      IndexType     iHiZ,
      float         *Z,
      IndexType     ldZ)
{
    CXXLAPACK_DEBUG_OUT("slahqr");

    IndexType info;
    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(slahqr)(&wantT_,
                        &wantZ_,
                        &n,
                        &iLo,
                        &iHi,
                        H,
                        &ldH,
                        wr,
                        wi,
                        &iLoZ,
                        &iHiZ,
                        Z,
                        &ldZ,
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
lahqr(bool          wantT,
      bool          wantZ,
      IndexType     n,
      IndexType     iLo,
      IndexType     iHi,
      double        *H,
      IndexType     ldH,
      double        *wr,
      double        *wi,
      IndexType     iLoZ,
      IndexType     iHiZ,
      double        *Z,
      IndexType     ldZ)
{
    CXXLAPACK_DEBUG_OUT("dlahqr");

    IndexType info;
    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(dlahqr)(&wantT_,
                        &wantZ_,
                        &n,
                        &iLo,
                        &iHi,
                        H,
                        &ldH,
                        wr,
                        wi,
                        &iLoZ,
                        &iHiZ,
                        Z,
                        &ldZ,
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
lahqr(bool                      wantT,
      bool                      wantZ,
      IndexType                 n,
      IndexType                 iLo,
      IndexType                 iHi,
      std::complex<float >      *H,
      IndexType                 ldH,
      std::complex<float >      *w,
      IndexType                 iLoZ,
      IndexType                 iHiZ,
      std::complex<float >      *Z,
      IndexType                 ldZ)
{
    CXXLAPACK_DEBUG_OUT("clahqr");

    IndexType info;
    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(clahqr)(&wantT_,
                        &wantZ_,
                        &n,
                        &iLo,
                        &iHi,
                        reinterpret_cast<float  *>(H),
                        &ldH,
                        reinterpret_cast<float  *>(w),
                        &iLoZ,
                        &iHiZ,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
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
lahqr(bool                      wantT,
      bool                      wantZ,
      IndexType                 n,
      IndexType                 iLo,
      IndexType                 iHi,
      std::complex<double>      *H,
      IndexType                 ldH,
      std::complex<double>      *w,
      IndexType                 iLoZ,
      IndexType                 iHiZ,
      std::complex<double>      *Z,
      IndexType                 ldZ)
{
    CXXLAPACK_DEBUG_OUT("zlahqr");

    IndexType info;
    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(zlahqr)(&wantT_,
                        &wantZ_,
                        &n,
                        &iLo,
                        &iHi,
                        reinterpret_cast<double *>(H),
                        &ldH,
                        reinterpret_cast<double *>(w),
                        &iLoZ,
                        &iHiZ,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
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

#endif // CXXLAPACK_INTERFACE_LAHQR_TCC

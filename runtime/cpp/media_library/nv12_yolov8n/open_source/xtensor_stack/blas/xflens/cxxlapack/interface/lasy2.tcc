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

#ifndef CXXLAPACK_INTERFACE_LASY2_TCC
#define CXXLAPACK_INTERFACE_LASY2_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
lasy2(bool          transTL,
      bool          transTR,
      IndexType     sgn,
      IndexType     n1,
      IndexType     n2,
      const float   *TL,
      IndexType     ldTL,
      const float   *TR,
      IndexType     ldTR,
      const float   *B,
      IndexType     ldB,
      float         &scale,
      float         *X,
      IndexType     ldX,
      float         &xNorm)
{
    CXXLAPACK_DEBUG_OUT("slasy2");

    IndexType info;
    IndexType transTL_ = transTL;
    IndexType transTR_ = transTR;
    LAPACK_IMPL(slasy2)(&transTL_,
                        &transTR_,
                        &sgn,
                        &n1,
                        &n2,
                        TL,
                        &ldTL,
                        TR,
                        &ldTR,
                        B,
                        &ldB,
                        &scale,
                        X,
                        &ldX,
                        &xNorm,
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
lasy2(bool          transTL,
      bool          transTR,
      IndexType     sgn,
      IndexType     n1,
      IndexType     n2,
      const double  *TL,
      IndexType     ldTL,
      const double  *TR,
      IndexType     ldTR,
      const double  *B,
      IndexType     ldB,
      double        &scale,
      double        *X,
      IndexType     ldX,
      double        &xNorm)
{
    CXXLAPACK_DEBUG_OUT("dlasy2");

    IndexType info;
    IndexType transTL_ = transTL;
    IndexType transTR_ = transTR;
    LAPACK_IMPL(dlasy2)(&transTL_,
                        &transTR_,
                        &sgn,
                        &n1,
                        &n2,
                        TL,
                        &ldTL,
                        TR,
                        &ldTR,
                        B,
                        &ldB,
                        &scale,
                        X,
                        &ldX,
                        &xNorm,
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

#endif // CXXLAPACK_INTERFACE_LASY2_TCC

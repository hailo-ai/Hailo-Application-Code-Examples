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

#ifndef CXXLAPACK_INTERFACE_LAEIN_TCC
#define CXXLAPACK_INTERFACE_LAEIN_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
laein(bool                  rightv,
      bool                  noinit,
      IndexType             n,
      const float           *H,
      IndexType             ldH,
      const float           wr,
      const float           wi,
      float                 *vr,
      float                 *vi,
      float                 *B,
      IndexType             ldB,
      float                 *work,
      float                 eps3,
      float                 smlnum,
      float                 bignum)
{
    CXXLAPACK_DEBUG_OUT("slaein");

    IndexType info;
    IndexType rightv_ = rightv;
    IndexType noinit_ = noinit;
    LAPACK_IMPL(slaein)(&rightv_,
                        &noinit_,
                        &n,
                        H,
                        &ldH,
                        &wr,
                        &wi,
                        vr,
                        vi,
                        B,
                        &ldB,
                        work,
                        &eps3,
                        &smlnum,
                        &bignum,
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
laein(bool                  rightv,
      bool                  noinit,
      IndexType             n,
      const double          *H,
      IndexType             ldH,
      const double          wr,
      const double          wi,
      double                *vr,
      double                *vi,
      double                *B,
      IndexType             ldB,
      double                *work,
      double                eps3,
      double                smlnum,
      double                bignum)
{
    CXXLAPACK_DEBUG_OUT("dlaein");

    IndexType info;
    IndexType rightv_ = rightv;
    IndexType noinit_ = noinit;
    LAPACK_IMPL(dlaein)(&rightv_,
                        &noinit_,
                        &n,
                        H,
                        &ldH,
                        &wr,
                        &wi,
                        vr,
                        vi,
                        B,
                        &ldB,
                        work,
                        &eps3,
                        &smlnum,
                        &bignum,
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
laein(bool                        rightv,
      bool                        noinit,
      IndexType                   n,
      const std::complex<float >  *H,
      IndexType                   ldH,
      const std::complex<float >  w,
      std::complex<float >        *v,
      std::complex<float >        *B,
      IndexType                   ldB,
      float                       *rWork,
      float                       eps3,
      float                       smlnum,
      float                       bignum)
{
    CXXLAPACK_DEBUG_OUT("claein");

    IndexType info;
    IndexType rightv_ = rightv;
    IndexType noinit_ = noinit;
    LAPACK_IMPL(claein)(&rightv_,
                        &noinit_,
                        &n,
                        reinterpret_cast<const float  *>(H),
                        &ldH,
                        reinterpret_cast<const float  *>(&w),
                        reinterpret_cast<float  *>(v),
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        rWork,
                        &eps3,
                        &smlnum,
                        &bignum,
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
laein(bool                        rightv,
      bool                        noinit,
      IndexType                   n,
      const std::complex<double>  *H,
      IndexType                   ldH,
      const std::complex<double>  w,
      std::complex<double>        *v,
      std::complex<double>        *B,
      IndexType                   ldB,
      double                      *rWork,
      double                      eps3,
      double                      smlnum,
      double                      bignum)
{
    CXXLAPACK_DEBUG_OUT("zlaein");

    IndexType info;
    IndexType rightv_ = rightv;
    IndexType noinit_ = noinit;
    LAPACK_IMPL(zlaein)(&rightv_,
                        &noinit_,
                        &n,
                        reinterpret_cast<const double *>(H),
                        &ldH,
                        reinterpret_cast<const double *>(&w),
                        reinterpret_cast<double *>(v),
                        reinterpret_cast<double *>(B),
                        &ldB,
                        rWork,
                        &eps3,
                        &smlnum,
                        &bignum,
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

#endif // CXXLAPACK_INTERFACE_LAEIN_TCC

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

#ifndef CXXLAPACK_INTERFACE_GEEVX_TCC
#define CXXLAPACK_INTERFACE_GEEVX_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
geevx(char          balanc,
      char          jobVL,
      char          jobVR,
      char          sense,
      IndexType     n,
      float         *A,
      IndexType     ldA,
      float         *wr,
      float         *wi,
      float         *VL,
      IndexType     ldVL,
      float         *VR,
      IndexType     ldVR,
      IndexType     &iLo,
      IndexType     &iHi,
      float         *scale,
      float         &ABnorm,
      float         *rCondE,
      float         *rCondV,
      float         *work,
      IndexType     lWork,
      IndexType     *iWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgeevx");
    LAPACK_IMPL(sgeevx)(&balanc,
                        &jobVL,
                        &jobVR,
                        &sense,
                        &n,
                        A,
                        &ldA,
                        wr,
                        wi,
                        VL,
                        &ldVL,
                        VR,
                        &ldVR,
                        &iLo,
                        &iHi,
                        scale,
                        &ABnorm,
                        rCondE,
                        rCondV,
                        work,
                        &lWork,
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
geevx(char          balanc,
      char          jobVL,
      char          jobVR,
      char          sense,
      IndexType     n,
      double        *A,
      IndexType     ldA,
      double        *wr,
      double        *wi,
      double        *VL,
      IndexType     ldVL,
      double        *VR,
      IndexType     ldVR,
      IndexType     &iLo,
      IndexType     &iHi,
      double        *scale,
      double        &ABnorm,
      double        *rCondE,
      double        *rCondV,
      double        *work,
      IndexType     lWork,
      IndexType     *iWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgeevx");
    LAPACK_IMPL(dgeevx)(&balanc,
                        &jobVL,
                        &jobVR,
                        &sense,
                        &n,
                        A,
                        &ldA,
                        wr,
                        wi,
                        VL,
                        &ldVL,
                        VR,
                        &ldVR,
                        &iLo,
                        &iHi,
                        scale,
                        &ABnorm,
                        rCondE,
                        rCondV,
                        work,
                        &lWork,
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
geevx(char                    balanc,
      char                    jobVL,
      char                    jobVR,
      char                    sense,
      IndexType               n,
      std::complex<float >    *A,
      IndexType               ldA,
      std::complex<float >    *w,
      std::complex<float >    *VL,
      IndexType               ldVL,
      std::complex<float >    *VR,
      IndexType               ldVR,
      IndexType               &iLo,
      IndexType               &iHi,
      float                   *scale,
      float                   &ABnorm,
      float                   *rCondE,
      float                   *rCondV,
      std::complex<float >    *work,
      IndexType               lWork,
      float                   *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgeevx");
    LAPACK_IMPL(cgeevx)(&balanc,
                        &jobVL,
                        &jobVR,
                        &sense,
                        &n,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(w),
                        reinterpret_cast<float  *>(VL),
                        &ldVL,
                        reinterpret_cast<float  *>(VR),
                        &ldVR,
                        &iLo,
                        &iHi,
                        scale,
                        &ABnorm,
                        rCondE,
                        rCondV,
                        reinterpret_cast<float  *>(work),
                        &lWork,
                        rWork,
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
geevx(char                    balanc,
      char                    jobVL,
      char                    jobVR,
      char                    sense,
      IndexType               n,
      std::complex<double>    *A,
      IndexType               ldA,
      std::complex<double>    *w,
      std::complex<double>    *VL,
      IndexType               ldVL,
      std::complex<double>    *VR,
      IndexType               ldVR,
      IndexType               &iLo,
      IndexType               &iHi,
      double                  *scale,
      double                  &ABnorm,
      double                  *rCondE,
      double                  *rCondV,
      std::complex<double>    *work,
      IndexType               lWork,
      double                  *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgeevx");
    LAPACK_IMPL(zgeevx)(&balanc,
                        &jobVL,
                        &jobVR,
                        &sense,
                        &n,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(w),
                        reinterpret_cast<double *>(VL),
                        &ldVL,
                        reinterpret_cast<double *>(VR),
                        &ldVR,
                        &iLo,
                        &iHi,
                        scale,
                        &ABnorm,
                        rCondE,
                        rCondV,
                        reinterpret_cast<double *>(work),
                        &lWork,
                        rWork,
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

#endif // CXXLAPACK_INTERFACE_GEEVX_TCC

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

#ifndef CXXLAPACK_INTERFACE_GGEVX_TCC
#define CXXLAPACK_INTERFACE_GGEVX_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
ggevx(char                  balanc,
      char                  jobVL,
      char                  jobvr,
      char                  sense,
      IndexType             n,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      float                 *alphar,
      float                 *alphai,
      float                 *beta,
      float                 *VL,
      IndexType             ldVL,
      float                 *VR,
      IndexType             ldVR,
      IndexType             &ilo,
      IndexType             &ihi,
      float                 *lscale,
      float                 *rscale,
      float                 &abnrm,
      float                 &bbnrm,
      float                 *rConde,
      float                 *rCondv,
      float                 *work,
      IndexType             lWork,
      IndexType             *iWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("sggevx");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(sggevx)(&balanc,
                        &jobVL,
                        &jobvr,
                        &sense,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        alphar,
                        alphai,
                        beta,
                        VL,
                        &ldVL,
                        VR,
                        &ldVR,
                        &ilo,
                        &ihi,
                        lscale,
                        rscale,
                        &abnrm,
                        &bbnrm,
                        rConde,
                        rCondv,
                        work,
                        &lWork,
                        iWork,
                        bwork,
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
ggevx(char                  balanc,
      char                  jobVL,
      char                  jobvr,
      char                  sense,
      IndexType             n,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      double                *alphar,
      double                *alphai,
      double                *beta,
      double                *VL,
      IndexType             ldVL,
      double                *VR,
      IndexType             ldVR,
      IndexType             &ilo,
      IndexType             &ihi,
      double                *lscale,
      double                *rscale,
      double                &abnrm,
      double                &bbnrm,
      double                *rConde,
      double                *rCondv,
      double                *work,
      IndexType             lWork,
      IndexType             *iWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("dggevx");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(dggevx)(&balanc,
                        &jobVL,
                        &jobvr,
                        &sense,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        alphar,
                        alphai,
                        beta,
                        VL,
                        &ldVL,
                        VR,
                        &ldVR,
                        &ilo,
                        &ihi,
                        lscale,
                        rscale,
                        &abnrm,
                        &bbnrm,
                        rConde,
                        rCondv,
                        work,
                        &lWork,
                        iWork,
                        bwork,
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
ggevx(char                  balanc,
      char                  jobVL,
      char                  jobvr,
      char                  sense,
      IndexType             n,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      std::complex<float >  *alpha,
      std::complex<float >  *beta,
      std::complex<float >  *VL,
      IndexType             ldVL,
      std::complex<float >  *VR,
      IndexType             ldVR,
      IndexType             &ilo,
      IndexType             &ihi,
      float                 *lscale,
      float                 *rscale,
      float                 &abnrm,
      float                 &bbnrm,
      float                 *rConde,
      float                 *rCondv,
      std::complex<float >  *work,
      IndexType             lWork,
      float                 *rWork,
      IndexType             *iWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("cggevx");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(cggevx)(&balanc,
                        &jobVL,
                        &jobvr,
                        &sense,
                        &n,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(alpha),
                        reinterpret_cast<float  *>(beta),
                        reinterpret_cast<float  *>(VL),
                        &ldVL,
                        reinterpret_cast<float  *>(VR),
                        &ldVR,
                        &ilo,
                        &ihi,
                        lscale,
                        rscale,
                        &abnrm,
                        &bbnrm,
                        rConde,
                        rCondv,
                        reinterpret_cast<float  *>(work),
                        &lWork,
                        rWork,
                        iWork,
                        bwork,
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
ggevx(char                  balanc,
      char                  jobVL,
      char                  jobvr,
      char                  sense,
      IndexType             n,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      std::complex<double>  *alpha,
      std::complex<double>  *beta,
      std::complex<double>  *VL,
      IndexType             ldVL,
      std::complex<double>  *VR,
      IndexType             ldVR,
      IndexType             &ilo,
      IndexType             &ihi,
      double                *lscale,
      double                *rscale,
      double                &abnrm,
      double                &bbnrm,
      double                *rConde,
      double                *rCondv,
      std::complex<double>  *work,
      IndexType             lWork,
      double                *rWork,
      IndexType             *iWork,
      bool                  *bwork)
{

    CXXLAPACK_DEBUG_OUT("zggevx");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(zggevx)(&balanc,
                        &jobVL,
                        &jobvr,
                        &sense,
                        &n,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(alpha),
                        reinterpret_cast<double *>(beta),
                        reinterpret_cast<double *>(VL),
                        &ldVL,
                        reinterpret_cast<double *>(VR),
                        &ldVR,
                        &ilo,
                        &ihi,
                        lscale,
                        rscale,
                        &abnrm,
                        &bbnrm,
                        rConde,
                        rCondv,
                        reinterpret_cast<double *>(work),
                        &lWork,
                        rWork,
                        iWork,
                        bwork,
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

#endif // CXXLAPACK_INTERFACE_GGEVX_TCC

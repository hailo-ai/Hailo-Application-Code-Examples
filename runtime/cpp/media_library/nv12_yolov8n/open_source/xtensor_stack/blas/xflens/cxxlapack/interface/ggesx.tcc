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

#ifndef CXXLAPACK_INTERFACE_GGESX_TCC
#define CXXLAPACK_INTERFACE_GGESX_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
ggesx(char                  jobvsl,
      char                  jobvsr,
      char                  sort,
      bool                  selctg,
      IndexType             n,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      IndexType             &sdim,
      float                 *alpha,
      float                 *beta,
      float                 *Vsl,
      IndexType             ldVsl,
      float                 *Vsr,
      IndexType             ldVsr,
      float                 *rConde,
      float                 *rCondv,
      float                 *work,
      IndexType             lWork,
      IndexType             *iWork,
      IndexType             liWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("sggesx");

    IndexType info;
    IndexType selctg_ = selctg;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(sggesx)(&jobvsl,
                        &jobvsr,
                        &sort,
                        &selctg_,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        &sdim,
                        alpha,
                        beta,
                        Vsl,
                        &ldVsl,
                        Vsr,
                        &ldVsr,
                        rConde,
                        rCondv,
                        work,
                        &lWork,
                        iWork,
                        liWork,
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
ggesx(char                  jobvsl,
      char                  jobvsr,
      char                  sort,
      bool                  selctg,
      IndexType             n,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      IndexType             &sdim,
      double                *alpha,
      double                *beta,
      double                *Vsl,
      IndexType             ldVsl,
      double                *Vsr,
      IndexType             ldVsr,
      double                *rConde,
      double                *rCondv,
      double                *work,
      IndexType             lWork,
      IndexType             *iWork,
      IndexType             liWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("dggesx");

    IndexType info;
    IndexType selctg_ = selctg;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(dggesx)(&jobvsl,
                        &jobvsr,
                        &sort,
                        &selctg_,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        &sdim,
                        alpha,
                        beta,
                        Vsl,
                        &ldVsl,
                        Vsr,
                        &ldVsr,
                        rConde,
                        rCondv,
                        work,
                        &lWork,
                        iWork,
                        liWork,
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
ggesx(char                  jobvsl,
      char                  jobvsr,
      char                  sort,
      bool                  selctg,
      IndexType             n,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      IndexType             &sdim,
      std::complex<float >  *alpha,
      std::complex<float >  *beta,
      std::complex<float >  *Vsl,
      IndexType             ldVsl,
      std::complex<float >  *Vsr,
      IndexType             ldVsr,
      float                 *rConde,
      float                 *rCondv,
      std::complex<float >  *work,
      IndexType             lWork,
      float                 rWork,
      IndexType             *iWork,
      IndexType             liWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("cggesx");

    IndexType info;
    IndexType selctg_ = selctg;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(cggesx)(&jobvsl,
                        &jobvsr,
                        &sort,
                        &selctg_,
                        &n,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        &sdim,
                        reinterpret_cast<float  *>(alpha),
                        reinterpret_cast<float  *>(beta),
                        reinterpret_cast<float  *>(Vsl),
                        &ldVsl,
                        reinterpret_cast<float  *>(Vsr),
                        &ldVsr,
                        rConde,
                        rCondv,
                        reinterpret_cast<float  *>(work),
                        &lWork,
                        rWork,
                        iWork,
                        &liWork,
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
ggesx(char                  jobvsl,
      char                  jobvsr,
      char                  sort,
      bool                  selctg,
      IndexType             n,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      IndexType             &sdim,
      std::complex<double>  *alpha,
      std::complex<double>  *beta,
      std::complex<double>  *Vsl,
      IndexType             ldVsl,
      std::complex<double>  *Vsr,
      IndexType             ldVsr,
      double                *rConde,
      double                *rCondv,
      std::complex<double>  *work,
      IndexType             lWork,
      double                rWork,
      IndexType             *iWork,
      IndexType             liWork,
      bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("zggesx");

    IndexType info;
    IndexType selctg_ = selctg;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(zggesx)(&jobvsl,
                        &jobvsr,
                        &sort,
                        &selctg_,
                        &n,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        &sdim,
                        reinterpret_cast<double *>(alpha),
                        reinterpret_cast<double *>(beta),
                        reinterpret_cast<double *>(Vsl),
                        &ldVsl,
                        reinterpret_cast<double *>(Vsr),
                        &ldVsr,
                        rConde,
                        rCondv,
                        reinterpret_cast<double *>(work),
                        &lWork,
                        rWork,
                        iWork,
                        &liWork,
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

#endif // CXXLAPACK_INTERFACE_GGESX_TCC

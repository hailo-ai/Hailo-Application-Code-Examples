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

#ifndef CXXLAPACK_INTERFACE_GEGS_TCC
#define CXXLAPACK_INTERFACE_GEGS_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gegs(char                  jobvsl,
     char                  jobvsr,
     IndexType             n,
     float                 *A,
     IndexType             ldA,
     float                 *B,
     IndexType             ldB,
     float                 *alpha,
     float                 *beta,
     float                 *Vsl,
     IndexType             ldVsl,
     float                 *Vsr,
     IndexType             ldVsr,
     float                 *work,
     IndexType             lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgegs");
    LAPACK_IMPL(sgegs)(&jobvsl,
                       &jobvsr,
                       &n,
                       A,
                       &ldA,
                       B,
                       &ldB,
                       alpha,
                       beta,
                       Vsl,
                       &ldVsl,
                       Vsr,
                       &ldVsr,
                       work,
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
gegs(char                  jobvsl,
     char                  jobvsr,
     IndexType             n,
     double                *A,
     IndexType             ldA,
     double                *B,
     IndexType             ldB,
     double                *alpha,
     double                *beta,
     double                *Vsl,
     IndexType             ldVsl,
     double                *Vsr,
     IndexType             ldVsr,
     double                *work,
     IndexType             lWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgegs");
    LAPACK_IMPL(dgegs)(&jobvsl,
                       &jobvsr,
                       &n,
                       A,
                       &ldA,
                       B,
                       &ldB,
                       alpha,
                       beta,
                       Vsl,
                       &ldVsl,
                       Vsr,
                       &ldVsr,
                       work,
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
gegs(char                  jobvsl,
     char                  jobvsr,
     IndexType             n,
     std::complex<float >  *A,
     IndexType             ldA,
     std::complex<float >  *B,
     IndexType             ldB,
     std::complex<float >  *alpha,
     std::complex<float >  *beta,
     std::complex<float >  *Vsl,
     IndexType             ldVsl,
     std::complex<float >  *Vsr,
     IndexType             ldVsr,
     std::complex<float >  *work,
     IndexType             lWork,
     double                *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgegs");
    LAPACK_IMPL(cgegs)(&jobvsl,
                       &jobvsr,
                       &n,
                       reinterpret_cast<float  *>(A),
                       &ldA,
                       reinterpret_cast<float  *>(B),
                       &ldB,
                       reinterpret_cast<float  *>(alpha),
                       reinterpret_cast<float  *>(beta),
                       reinterpret_cast<float  *>(Vsl),
                       &ldVsl,
                       reinterpret_cast<float  *>(Vsr),
                       &ldVsr,
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
gegs(char                  jobvsl,
     char                  jobvsr,
     IndexType             n,
     std::complex<double>  *A,
     IndexType             ldA,
     std::complex<double>  *B,
     IndexType             ldB,
     std::complex<double>  *alpha,
     std::complex<double>  *beta,
     std::complex<double>  *Vsl,
     IndexType             ldVsl,
     std::complex<double>  *Vsr,
     IndexType             ldVsr,
     std::complex<double>  *work,
     IndexType             lWork,
     double                *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgegs");
    LAPACK_IMPL(zgegs)(&jobvsl,
                       &jobvsr,
                       &n,
                       reinterpret_cast<double *>(A),
                       &ldA,
                       reinterpret_cast<double *>(B),
                       &ldB,
                       reinterpret_cast<double *>(alpha),
                       reinterpret_cast<double *>(beta),
                       reinterpret_cast<double *>(Vsl),
                       &ldVsl,
                       reinterpret_cast<double *>(Vsr),
                       &ldVsr,
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

#endif // CXXLAPACK_INTERFACE_GEGS_TCC

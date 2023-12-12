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

#ifndef CXXLAPACK_INTERFACE_GGES_TCC
#define CXXLAPACK_INTERFACE_GGES_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gges(char                  jobvsl,
     char                  jobvsr,
     char                  sort,
     IndexType             (*select)(const float  *, const float  *),
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
     float                 *work,
     IndexType             lWork,
     bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("sgges");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(sgges)(&jobvsl,
                       &jobvsr,
                       &sort,
                       select,
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
                       work,
                       &lWork,
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
gges(char                  jobvsl,
     char                  jobvsr,
     char                  sort,
     IndexType             (*select)(const double *, const double *),
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
     double                *work,
     IndexType             lWork,
     bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("dgges");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(dgges)(&jobvsl,
                       &jobvsr,
                       &sort,
                       select,
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
                       work,
                       &lWork,
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
gges(char                  jobvsl,
     char                  jobvsr,
     char                  sort,
     IndexType              (*select)(const std::complex<float > *),
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
     std::complex<float >  *work,
     IndexType             lWork,
     double                rWork,
     bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("cgges");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(cgges)(&jobvsl,
                       &jobvsr,
                       &sort,
                       &select,
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
                       reinterpret_cast<float  *>(work),
                       &lWork,
                       rWork,
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
gges(char                  jobvsl,
     char                  jobvsr,
     char                  sort,
     IndexType              (*select)(const std::complex<double> *),
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
     std::complex<double>  *work,
     IndexType             lWork,
     double                rWork,
     bool                  *bwork)
{
    CXXLAPACK_DEBUG_OUT("zgges");

    IndexType info;
    // TODO: Convert bwork into a logical array!
    LAPACK_IMPL(zgges)(&jobvsl,
                       &jobvsr,
                       &sort,
                       &select,
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
                       reinterpret_cast<double *>(work),
                       &lWork,
                       rWork,
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

#endif // CXXLAPACK_INTERFACE_GGES_TCC

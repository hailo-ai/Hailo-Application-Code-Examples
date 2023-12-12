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

#ifndef CXXLAPACK_INTERFACE_GEES_TCC
#define CXXLAPACK_INTERFACE_GEES_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gees(char           jobVS,
     char           sort,
     IndexType      (*select)(const float *, const float *),
     IndexType      n,
     float          *A,
     IndexType      ldA,
     IndexType      &sdim,
     float          *wr,
     float          *wi,
     float          *VS,
     IndexType      ldVS,
     float          *work,
     IndexType      lWork,
     IndexType      *bWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgees");
    LAPACK_IMPL(sgees)(&jobVS,
                       &sort,
                       select,
                       &n,
                       A,
                       &ldA,
                       &sdim,
                       wr,
                       wi,
                       VS,
                       &ldVS,
                       work,
                       &lWork,
                       bWork,
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
gees(char           jobVS,
     char           sort,
     IndexType      (*select)(const double *, const double *),
     IndexType      n,
     double         *A,
     IndexType      ldA,
     IndexType      &sdim,
     double         *wr,
     double         *wi,
     double         *VS,
     IndexType      ldVS,
     double         *work,
     IndexType      lWork,
     IndexType      *bWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgees");
    LAPACK_IMPL(dgees)(&jobVS,
                       &sort,
                       select,
                       &n,
                       A,
                       &ldA,
                       &sdim,
                       wr,
                       wi,
                       VS,
                       &ldVS,
                       work,
                       &lWork,
                       bWork,
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
gees(char                   jobVS,
     char                   sort,
     IndexType              (*select)(const std::complex<float > *),
     IndexType              n,
     std::complex<float >   *A,
     const IndexType        ldA,
     IndexType              &sdim,
     std::complex<float >   *w,
     std::complex<float >   *VS,
     const IndexType        ldVS,
     std::complex<float >   *work,
     const IndexType        lWork,
     float                  *rWork,
     IndexType              *bWork)
{
    typedef IndexType (*LapackSelect)(const float  *);

    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgees");
    LAPACK_IMPL(cgees)(&jobVS,
                       &sort,
                       reinterpret_cast<LapackSelect>(select),
                       &n,
                       reinterpret_cast<float  *>(A),
                       &ldA,
                       &sdim,
                       reinterpret_cast<float  *>(w),
                       reinterpret_cast<float  *>(VS),
                       &ldVS,
                       reinterpret_cast<float  *>(work),
                       &lWork,
                       rWork,
                       bWork,
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
gees(char                   jobVS,
     char                   sort,
     IndexType              (*select)(const std::complex<double> *),
     IndexType              n,
     std::complex<double>   *A,
     const IndexType        ldA,
     IndexType              &sdim,
     std::complex<double>   *w,
     std::complex<double>   *VS,
     const IndexType        ldVS,
     std::complex<double>   *work,
     const IndexType        lWork,
     double                 *rWork,
     IndexType              *bWork)
{
    typedef IndexType (*LapackSelect)(const double *);

    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgees");
    LAPACK_IMPL(zgees)(&jobVS,
                       &sort,
                       reinterpret_cast<LapackSelect>(select),
                       &n,
                       reinterpret_cast<double *>(A),
                       &ldA,
                       &sdim,
                       reinterpret_cast<double *>(w),
                       reinterpret_cast<double *>(VS),
                       &ldVS,
                       reinterpret_cast<double *>(work),
                       &lWork,
                       rWork,
                       bWork,
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

#endif // CXXLAPACK_INTERFACE_GEES_TCC

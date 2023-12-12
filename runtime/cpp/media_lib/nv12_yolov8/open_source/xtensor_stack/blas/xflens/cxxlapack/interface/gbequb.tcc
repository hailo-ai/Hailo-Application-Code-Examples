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

#ifndef CXXLAPACK_INTERFACE_GBEQUB_TCC
#define CXXLAPACK_INTERFACE_GBEQUB_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gbequb(IndexType             m,
       IndexType             n,
       IndexType             kl,
       IndexType             ku,
       const float           *Ab,
       IndexType             ldAb,
       float                 *r,
       float                 *c,
       float                 &rowcnd,
       float                 &colcnd,
       float                 &amax)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgbequb");
    LAPACK_IMPL(sgbequb)(&m,
                         &n,
                         &kl,
                         &ku,
                         Ab,
                         &ldAb,
                         r,
                         c,
                         &rowcnd,
                         &colcnd,
                         &amax,
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
gbequb(IndexType             m,
       IndexType             n,
       IndexType             kl,
       IndexType             ku,
       const double          *Ab,
       IndexType             ldAb,
       double                *r,
       double                *c,
       double                &rowcnd,
       double                &colcnd,
       double                &amax)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgbequb");
    LAPACK_IMPL(dgbequb)(&m,
                         &n,
                         &kl,
                         &ku,
                         Ab,
                         &ldAb,
                         r,
                         c,
                         &rowcnd,
                         &colcnd,
                         &amax,
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
gbequb(IndexType                   m,
       IndexType                   n,
       IndexType                   kl,
       IndexType                   ku,
       const std::complex<float >  *Ab,
       IndexType                   ldAb,
       float                       *r,
       float                       *c,
       float                       &rowcnd,
       float                       &colcnd,
       float                       &amax)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgbequb");
    LAPACK_IMPL(cgbequb)(&m,
                         &n,
                         &kl,
                         &ku,
                         reinterpret_cast<const float  *>(Ab),
                         &ldAb,
                         r,
                         c,
                         &rowcnd,
                         &colcnd,
                         &amax,
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
gbequb(IndexType                   m,
       IndexType                   n,
       IndexType                   kl,
       IndexType                   ku,
       const std::complex<double>  *Ab,
       IndexType                   ldAb,
       double                      *r,
       double                      *c,
       double                      &rowcnd,
       double                      &colcnd,
       double                      &amax)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgbequb");
    LAPACK_IMPL(zgbequb)(&m,
                         &n,
                         &kl,
                         &ku,
                         reinterpret_cast<const double *>(Ab),
                         &ldAb,
                         r,
                         c,
                         &rowcnd,
                         &colcnd,
                         &amax,
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

#endif // CXXLAPACK_INTERFACE_GBEQUB_TCC

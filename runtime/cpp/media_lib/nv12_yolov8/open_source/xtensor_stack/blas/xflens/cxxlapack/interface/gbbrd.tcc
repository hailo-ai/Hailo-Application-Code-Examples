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

#ifndef CXXLAPACK_INTERFACE_GBBRD_TCC
#define CXXLAPACK_INTERFACE_GBBRD_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gbbrd(char                  vect,
      IndexType             m,
      IndexType             n,
      IndexType             ncc,
      IndexType             kl,
      IndexType             ku,
      float                 *Ab,
      IndexType             ldAb,
      float                 *d,
      float                 *e,
      float                 *Q,
      IndexType             ldQ,
      float                 *PT,
      IndexType             ldPT,
      float                 *C,
      IndexType             ldC,
      float                 *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgbbrd");
    LAPACK_IMPL(sgbbrd)(&vect,
                        &m,
                        &n,
                        &ncc,
                        &kl,
                        &ku,
                        Ab,
                        &ldAb,
                        d,
                        e,
                        Q,
                        &ldQ,
                        PT,
                        &ldPT,
                        C,
                        &ldC,
                        work,
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
gbbrd(char                  vect,
      IndexType             m,
      IndexType             n,
      IndexType             ncc,
      IndexType             kl,
      IndexType             ku,
      double                *Ab,
      IndexType             ldAb,
      double                *d,
      double                *e,
      double                *Q,
      IndexType             ldQ,
      double                *PT,
      IndexType             ldPT,
      double                *C,
      IndexType             ldC,
      double                *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgbbrd");
    LAPACK_IMPL(dgbbrd)(&vect,
                        &m,
                        &n,
                        &ncc,
                        &kl,
                        &ku,
                        Ab,
                        &ldAb,
                        d,
                        e,
                        Q,
                        &ldQ,
                        PT,
                        &ldPT,
                        C,
                        &ldC,
                        work,
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
gbbrd(char                  vect,
      IndexType             m,
      IndexType             n,
      IndexType             ncc,
      IndexType             kl,
      IndexType             ku,
      std::complex<float >  *Ab,
      IndexType             ldAb,
      float                 *d,
      float                 *e,
      std::complex<float >  *Q,
      IndexType             ldQ,
      std::complex<float >  *PT,
      IndexType             ldPT,
      std::complex<float >  *C,
      IndexType             ldC,
      std::complex<float >  *work,
      float                 *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgbbrd");
    LAPACK_IMPL(cgbbrd)(&vect,
                        &m,
                        &n,
                        &ncc,
                        &kl,
                        &ku,
                        reinterpret_cast<float  *>(Ab),
                        &ldAb,
                        d,
                        e,
                        reinterpret_cast<float  *>(Q),
                        &ldQ,
                        reinterpret_cast<float  *>(PT),
                        &ldPT,
                        reinterpret_cast<float  *>(C),
                        &ldC,
                        reinterpret_cast<float  *>(work),
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
gbbrd(char                  vect,
      IndexType             m,
      IndexType             n,
      IndexType             ncc,
      IndexType             kl,
      IndexType             ku,
      std::complex<double>  *Ab,
      IndexType             ldAb,
      double                *d,
      double                *e,
      std::complex<double>  *Q,
      IndexType             ldQ,
      std::complex<double>  *PT,
      IndexType             ldPT,
      std::complex<double>  *C,
      IndexType             ldC,
      std::complex<double>  *work,
      double                *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgbbrd");
    LAPACK_IMPL(zgbbrd)(&vect,
                        &m,
                        &n,
                        &ncc,
                        &kl,
                        &ku,
                        reinterpret_cast<double *>(Ab),
                        &ldAb,
                        d,
                        e,
                        reinterpret_cast<double *>(Q),
                        &ldQ,
                        reinterpret_cast<double *>(PT),
                        &ldPT,
                        reinterpret_cast<double *>(C),
                        &ldC,
                        reinterpret_cast<double *>(work),
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

#endif // CXXLAPACK_INTERFACE_GBBRD_TCC

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

#ifndef CXXLAPACK_INTERFACE_HBGST_TCC
#define CXXLAPACK_INTERFACE_HBGST_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
hbgst(char                       vect,
      char                       uplo,
      IndexType                  n,
      IndexType                  ka,
      IndexType                  kb,
      std::complex<float >       *Ab,
      IndexType                  ldAb,
      const std::complex<float > *Bb,
      IndexType                  ldBb,
      std::complex<float >       *X,
      IndexType                  ldX,
      std::complex<float >       *work,
      float                      *rWork)
{
    CXXLAPACK_DEBUG_OUT("chbgst");

    IndexType info;
    LAPACK_IMPL(chbgst)(&vect,
                        &uplo,
                        &n,
                        &ka,
                        &kb,
                        reinterpret_cast<float  *>(Ab),
                        &ldAb,
                        reinterpret_cast<const float  *>(Bb),
                        &ldBb,
                        reinterpret_cast<float  *>(X),
                        &ldX,
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
hbgst(char                       vect,
      char                       uplo,
      IndexType                  n,
      IndexType                  ka,
      IndexType                  kb,
      std::complex<double>       *Ab,
      IndexType                  ldAb,
      const std::complex<double> *Bb,
      IndexType                  ldBb,
      std::complex<double>       *X,
      IndexType                  ldX,
      std::complex<double>       *work,
      double                     *rWork)
{
    CXXLAPACK_DEBUG_OUT("zhbgst");

    IndexType info;
    LAPACK_IMPL(zhbgst)(&vect,
                        &uplo,
                        &n,
                        &ka,
                        &kb,
                        reinterpret_cast<double *>(Ab),
                        &ldAb,
                        reinterpret_cast<const double *>(Bb),
                        &ldBb,
                        reinterpret_cast<double *>(X),
                        &ldX,
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

#endif // CXXLAPACK_INTERFACE_HBGST_TCC

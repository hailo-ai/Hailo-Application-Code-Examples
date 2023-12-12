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

#ifndef CXXLAPACK_INTERFACE_GESVX_H
#define CXXLAPACK_INTERFACE_GESVX_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    gesvx(char          fact,
          char          trans,
          IndexType     n,
          IndexType     nRhs,
          float         *A,
          IndexType     ldA,
          float         *AF,
          IndexType     ldAF,
          IndexType     *iPiv,
          char          equed,
          float         *r,
          float         *c,
          float         *B,
          IndexType     ldB,
          float         *X,
          IndexType     ldX,
          float         &rCond,
          float         *fErr,
          float         *bErr,
          float         *work,
          IndexType     *iWork);

template <typename IndexType>
    IndexType
    gesvx(char          fact,
          char          trans,
          IndexType     n,
          IndexType     nRhs,
          double        *A,
          IndexType     ldA,
          double        *AF,
          IndexType     ldAF,
          IndexType     *iPiv,
          char          equed,
          double        *r,
          double        *c,
          double        *B,
          IndexType     ldB,
          double        *X,
          IndexType     ldX,
          double        &rCond,
          double        *fErr,
          double        *bErr,
          double        *work,
          IndexType     *iWork);

    template <typename IndexType>
    IndexType
    gesvx(char                  fact,
          char                  trans,
          IndexType             n,
          IndexType             nRhs,
          std::complex<float >  *A,
          IndexType             ldA,
          std::complex<float >  *AF,
          IndexType             ldAF,
          IndexType             *iPiv,
          char                  equed,
          float                 *r,
          float                 *c,
          std::complex<float >  *B,
          IndexType             ldB,
          std::complex<float >  *X,
          IndexType             ldX,
          float                 &rCond,
          float                 *fErr,
          float                 *bErr,
          std::complex<float >  *work,
          float                 *rWork);

template <typename IndexType>
    IndexType
    gesvx(char                  fact,
          char                  trans,
          IndexType             n,
          IndexType             nRhs,
          std::complex<double>  *A,
          IndexType             ldA,
          std::complex<double>  *AF,
          IndexType             ldAF,
          IndexType             *iPiv,
          char                  equed,
          double                *r,
          double                *c,
          std::complex<double>  *B,
          IndexType             ldB,
          std::complex<double>  *X,
          IndexType             ldX,
          double                &rCond,
          double                *fErr,
          double                *bErr,
          std::complex<double>  *work,
          double                *rWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_GESVX_H

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

#ifndef CXXLAPACK_INTERFACE_UNCSD_H
#define CXXLAPACK_INTERFACE_UNCSD_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    uncsd(char                         jobu1,
          char                         jobu2,
          char                         jobv1t,
          char                         jobv2t,
          char                         trans,
          char                         signs,
          IndexType                    m,
          IndexType                    p,
          IndexType                    q,
          const std::complex<float >   *X11,
          IndexType                    ldX11,
          const std::complex<float >   *X12,
          IndexType                    ldX12,
          const std::complex<float >   *X21,
          IndexType                    ldX21,
          const std::complex<float >   *X22,
          IndexType                    ldX22,
          float                        *theta,
          std::complex<float >         *U1,
          IndexType                    ldU1,
          std::complex<float >         *U2,
          IndexType                    ldU2,
          std::complex<float >         *V1t,
          IndexType                    ldV1t,
          std::complex<float >         *V2t,
          IndexType                    ldV2t,
          std::complex<float >         *work,
          IndexType                    lWork,
          float                        *rWork,
          IndexType                    lrWork,
          IndexType                    *iWork);

template <typename IndexType>
    IndexType
    uncsd(char                         jobu1,
          char                         jobu2,
          char                         jobv1t,
          char                         jobv2t,
          char                         trans,
          char                         signs,
          IndexType                    m,
          IndexType                    p,
          IndexType                    q,
          const std::complex<double>   *X11,
          IndexType                    ldX11,
          const std::complex<double>   *X12,
          IndexType                    ldX12,
          const std::complex<double>   *X21,
          IndexType                    ldX21,
          const std::complex<double>   *X22,
          IndexType                    ldX22,
          double                       *theta,
          std::complex<double>         *U1,
          IndexType                    ldU1,
          std::complex<double>         *U2,
          IndexType                    ldU2,
          std::complex<double>         *V1t,
          IndexType                    ldV1t,
          std::complex<double>         *V2t,
          IndexType                    ldV2t,
          std::complex<double>         *work,
          IndexType                    lWork,
          double                       *rWork,
          IndexType                    lrWork,
          IndexType                    *iWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_UNCSD_H

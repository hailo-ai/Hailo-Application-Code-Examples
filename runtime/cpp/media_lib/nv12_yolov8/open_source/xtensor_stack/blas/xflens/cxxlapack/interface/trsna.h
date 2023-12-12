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

#ifndef CXXLAPACK_INTERFACE_TRSNA_H
#define CXXLAPACK_INTERFACE_TRSNA_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    trsna(char              job,
          char              howMany,
          const IndexType   *select,
          IndexType         n,
          const float       *T,
          IndexType         ldT,
          const float       *VL,
          IndexType         ldVL,
          const float       *VR,
          IndexType         ldVR,
          float             *s,
          float             *sep,
          IndexType         mm,
          IndexType         &m,
          float             *work,
          IndexType         ldWork,
          IndexType         *iWork);


template <typename IndexType>
    IndexType
    trsna(char              job,
          char              howMany,
          const IndexType   *select,
          IndexType         n,
          const double      *T,
          IndexType         ldT,
          const double      *VL,
          IndexType         ldVL,
          const double      *VR,
          IndexType         ldVR,
          double            *s,
          double            *sep,
          IndexType         mm,
          IndexType         &m,
          double            *work,
          IndexType         ldWork,
          IndexType         *iWork);

template <typename IndexType>
    IndexType
    trsna(char                          job,
          char                          howMany,
          const IndexType               *select,
          const IndexType               n,
          const std::complex<float >    *T,
          const IndexType               ldT,
          const std::complex<float >    *VL,
          const IndexType               ldVL,
          const std::complex<float >    *VR,
          const IndexType               ldVR,
          float                         *s,
          float                         *sep,
          const IndexType               mm,
          IndexType                     &m,
          std::complex<float >          *work,
          const IndexType               ldWork,
          float                         *rWork);

template <typename IndexType>
    IndexType
    trsna(char                          job,
          char                          howMany,
          const IndexType               *select,
          const IndexType               n,
          const std::complex<double>    *T,
          const IndexType               ldT,
          const std::complex<double>    *VL,
          const IndexType               ldVL,
          const std::complex<double>    *VR,
          const IndexType               ldVR,
          double                        *s,
          double                        *sep,
          const IndexType               mm,
          IndexType                     &m,
          std::complex<double>          *work,
          const IndexType               ldWork,
          double                        *rWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_TRSNA_H

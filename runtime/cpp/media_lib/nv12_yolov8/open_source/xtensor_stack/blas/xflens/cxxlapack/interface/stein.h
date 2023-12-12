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

#ifndef CXXLAPACK_INTERFACE_STEIN_H
#define CXXLAPACK_INTERFACE_STEIN_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    stein(IndexType             n,
          const float           *d,
          const float           *e,
          IndexType             m,
          const float           *w,
          const IndexType       *iblock,
          const IndexType       *isplit,
          float                 *Z,
          IndexType             ldZ,
          float                 *work,
          IndexType             *iWork,
          IndexType             *ifail);

template <typename IndexType>
    IndexType
    stein(IndexType             n,
          const double          *d,
          const double          *e,
          IndexType             m,
          const double          *w,
          const IndexType       *iblock,
          const IndexType       *isplit,
          double                *Z,
          IndexType             ldZ,
          double                *work,
          IndexType             *iWork,
          IndexType             *ifail);

template <typename IndexType>
    IndexType
    stein(IndexType             n,
          const float           *d,
          const float           *e,
          IndexType             m,
          const float           *w,
          const IndexType       *iblock,
          const IndexType       *isplit,
          std::complex<float >  *Z,
          IndexType             ldZ,
          float                 *work,
          IndexType             *iWork,
          IndexType             *ifail);

template <typename IndexType>
    IndexType
    stein(IndexType             n,
          const double          *d,
          const double          *e,
          IndexType             m,
          const double          *w,
          const IndexType       *iblock,
          const IndexType       *isplit,
          std::complex<double>  *Z,
          IndexType             ldZ,
          double                *work,
          IndexType             *iWork,
          IndexType             *ifail);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_STEIN_H

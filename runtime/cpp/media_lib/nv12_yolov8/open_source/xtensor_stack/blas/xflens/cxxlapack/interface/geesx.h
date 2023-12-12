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

#ifndef CXXLAPACK_INTERFACE_GEESX_H
#define CXXLAPACK_INTERFACE_GEESX_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    geesx(char              jobVS,
          char              sort,
          IndexType         (*select)(const float *, const float  *),
          char              sense,
          IndexType         n,
          float             *A,
          IndexType         ldA,
          IndexType         &sDim,
          float             *wr,
          float             *wi,
          float             *VS,
          IndexType         ldVS,
          float             *rCondE,
          float             *rCondV,
          float             *work,
          IndexType         lWork,
          IndexType         *iWork,
          IndexType         liWork,
          IndexType         *bWork);

template <typename IndexType>
    IndexType
    geesx(char              jobVS,
          char              sort,
          IndexType         (*select)(const double *, const double *),
          char              sense,
          IndexType         n,
          double            *A,
          IndexType         ldA,
          IndexType         &sDim,
          double            *wr,
          double            *wi,
          double            *VS,
          IndexType         ldVS,
          double            *rCondE,
          double            *rCondV,
          double            *work,
          IndexType         lWork,
          IndexType         *iWork,
          IndexType         liWork,
          IndexType         *bWork);

template <typename IndexType>
    IndexType
    geesx(char                  jobVS,
          char                  sort,
          IndexType             (*select)(const std::complex<float > *),
          char                  sense,
          IndexType             n,
          std::complex<float >  *A,
          IndexType             ldA,
          IndexType             &sDim,
          std::complex<float >  *w,
          std::complex<float >  *VS,
          IndexType             ldVS,
          float                 *rCondE,
          float                 *rCondV,
          std::complex<float >  *work,
          IndexType             lWork,
          float                 *rWork,
          IndexType             *bWork);

template <typename IndexType>
    IndexType
    geesx(char                  jobVS,
          char                  sort,
          IndexType             (*select)(const std::complex<double> *),
          char                  sense,
          IndexType             n,
          std::complex<double>  *A,
          IndexType             ldA,
          IndexType             &sDim,
          std::complex<double>  *w,
          std::complex<double>  *VS,
          IndexType             ldVS,
          double                *rCondE,
          double                *rCondV,
          std::complex<double>  *work,
          IndexType             lWork,
          double                *rWork,
          IndexType             *bWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_GEESX_H

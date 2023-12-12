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

#ifndef CXXLAPACK_INTERFACE_LAGS2_H
#define CXXLAPACK_INTERFACE_LAGS2_H 1

#include <complex>

namespace cxxlapack {

template <typename XFLENS_VOID=void>
    void
    lags2(bool                  upper,
          float                 a1,
          float                 a2,
          float                 a3,
          float                 b1,
          float                 b2,
          float                 b3,
          float                 &csu,
          float                 &snu,
          float                 &csv,
          float                 &snv,
          float                 &csq,
          float                 &snq);

template <typename XFLENS_VOID=void>
    void
    lags2(bool                  upper,
          double                a1,
          double                a2,
          double                a3,
          double                b1,
          double                b2,
          double                b3,
          double                &csu,
          double                &snu,
          double                &csv,
          double                &snv,
          double                &csq,
          double                &snq);

template <typename XFLENS_VOID=void>
    void
    lags2(bool                  upper,
          float                 a1,
          std::complex<float >  a2,
          float                 a3,
          float                 b1,
          std::complex<float >  b2,
          float                 b3,
          float                 &csu,
          std::complex<float >  &snu,
          float                 &csv,
          std::complex<float >  &snv,
          float                 &csq,
          std::complex<float >  &snq);

template <typename XFLENS_VOID=void>
    void
    lags2(bool                  upper,
          double                a1,
          std::complex<double>  a2,
          double                a3,
          double                b1,
          std::complex<double>  b2,
          double                b3,
          double                &csu,
          std::complex<double>  &snu,
          double                &csv,
          std::complex<double>  &snv,
          double                &csq,
          std::complex<double>  &snq);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAGS2_H

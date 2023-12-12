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

#ifndef CXXLAPACK_INTERFACE_HSEIN_H
#define CXXLAPACK_INTERFACE_HSEIN_H 1

#include <complex>

namespace cxxlapack {

// template <typename IndexType>
//     IndexType
//     hsein(char                  side,
//           char                  eigsrc,
//           char                  initv,
//           bool                  *select,
//           IndexType             n,
//           const float           *H,
//           IndexType             ldH,
//           float                 *wr,
//           const float           *wi,
//           float                 *VL,
//           IndexType             ldVL,
//           float                 *VR,
//           IndexType             ldVR,
//           IndexType             mm,
//           IndexType             &m,
//           float                 *work,
//           IndexType             *ifaill,
//           IndexType             *ifailr);
//
// template <typename IndexType>
//     IndexType
//     hsein(char                  side,
//           char                  eigsrc,
//           char                  initv,
//           bool                  *select,
//           IndexType             n,
//           const double          *H,
//           IndexType             ldH,
//           double                *wr,
//           const double          *wi,
//           double                *VL,
//           IndexType             ldVL,
//           double                *VR,
//           IndexType             ldVR,
//           IndexType             mm,
//           IndexType             &m,
//           double                *work,
//           IndexType             *ifaill,
//           IndexType             *ifailr);
//
// template <typename IndexType>
//     IndexType
//     hsein(char                        side,
//           char                        eigsrc,
//           char                        initv,
//           bool                        *select,
//           IndexType                   n,
//           const std::complex<float >  *H,
//           IndexType                   ldH,
//           std::complex<float >        *w,
//           std::complex<float >        *VL,
//           IndexType                   ldVL,
//           std::complex<float >        *VR,
//           IndexType                   ldVR,
//           IndexType                   mm,
//           IndexType                   &m,
//           std::complex<float >        *work,
//           float                       *rWork,
//           IndexType                   *ifaill,
//           IndexType                   *ifailr);
//
// template <typename IndexType>
//     IndexType
//     hsein(char                        side,
//           char                        eigsrc,
//           char                        initv,
//           bool                        *select,
//           IndexType                   n,
//           const std::complex<double>  *H,
//           IndexType                   ldH,
//           std::complex<double>        *w,
//           std::complex<double>        *VL,
//           IndexType                   ldVL,
//           std::complex<double>        *VR,
//           IndexType                   ldVR,
//           IndexType                   mm,
//           IndexType                   &m,
//           std::complex<double>        *work,
//           double                      *rWork,
//           IndexType                   *ifaill,
//           IndexType                   *ifailr);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_HSEIN_H

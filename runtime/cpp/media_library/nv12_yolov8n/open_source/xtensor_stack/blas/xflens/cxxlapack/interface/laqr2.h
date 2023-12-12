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

#ifndef CXXLAPACK_INTERFACE_LAQR2_H
#define CXXLAPACK_INTERFACE_LAQR2_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    void
    laqr2(bool          wantT,
          bool          wantZ,
          IndexType     u,
          IndexType     kTop,
          IndexType     kBot,
          IndexType     nw,
          float         *H,
          IndexType     ldH,
          IndexType     iLoZ,
          IndexType     iHiZ,
          float         *Z,
          IndexType     ldZ,
          IndexType     &ns,
          IndexType     &nd,
          float         *sr,
          float         *si,
          float         *V,
          IndexType     ldV,
          IndexType     nh,
          float         *T,
          IndexType     ldT,
          IndexType     nv,
          float         *WV,
          IndexType     ldWV,
          float         *work,
          IndexType     lWork);

template <typename IndexType>
    void
    laqr2(bool          wantT,
          bool          wantZ,
          IndexType     u,
          IndexType     kTop,
          IndexType     kBot,
          IndexType     nw,
          double        *H,
          IndexType     ldH,
          IndexType     iLoZ,
          IndexType     iHiZ,
          double        *Z,
          IndexType     ldZ,
          IndexType     &ns,
          IndexType     &nd,
          double        *sr,
          double        *si,
          double        *V,
          IndexType     ldV,
          IndexType     nh,
          double        *T,
          IndexType     ldT,
          IndexType     nv,
          double        *WV,
          IndexType     ldWV,
          double        *work,
          IndexType     lWork);

template <typename IndexType>
    void
    laqr2(bool                  wantT,
          bool                  wantZ,
          IndexType             n,
          IndexType             kTop,
          IndexType             kBot,
          IndexType             nw,
          std::complex<float >  *H,
          IndexType             ldH,
          IndexType             iLoZ,
          IndexType             iHiZ,
          std::complex<float >  *Z,
          IndexType             ldZ,
          IndexType             &ns,
          IndexType             &nd,
          std::complex<float >  *sh,
          std::complex<float >  *V,
          IndexType             ldV,
          IndexType             nh,
          std::complex<float >  *T,
          IndexType             ldT,
          IndexType             nv,
          std::complex<float >  *WV,
          IndexType             ldWV,
          std::complex<float >  *work,
          IndexType             lWork);

template <typename IndexType>
    void
    laqr2(bool                  wantT,
          bool                  wantZ,
          IndexType             n,
          IndexType             kTop,
          IndexType             kBot,
          IndexType             nw,
          std::complex<double>  *H,
          IndexType             ldH,
          IndexType             iLoZ,
          IndexType             iHiZ,
          std::complex<double>  *Z,
          IndexType             ldZ,
          IndexType             &ns,
          IndexType             &nd,
          std::complex<double>  *sh,
          std::complex<double>  *V,
          IndexType             ldV,
          IndexType             nh,
          std::complex<double>  *T,
          IndexType             ldT,
          IndexType             nv,
          std::complex<double>  *WV,
          IndexType             ldWV,
          std::complex<double>  *work,
          IndexType             lWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQR2_H

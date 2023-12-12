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

#ifndef CXXLAPACK_INTERFACE_LAQR5_TCC
#define CXXLAPACK_INTERFACE_LAQR5_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqr5(bool          wantT,
      bool          wantZ,
      IndexType     kacc22,
      IndexType     n,
      IndexType     kTop,
      IndexType     kBot,
      IndexType     nShifts,
      float         *sr,
      float         *si,
      float         *H,
      IndexType     ldH,
      IndexType     iLoZ,
      IndexType     iHiZ,
      float         *Z,
      IndexType     ldZ,
      float         *V,
      IndexType     ldV,
      float         *U,
      IndexType     ldU,
      IndexType     nv,
      float         *WV,
      IndexType     ldWV,
      IndexType     nh,
      float         *WH,
      IndexType     ldWH)
{
    CXXLAPACK_DEBUG_OUT("slaqr5");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(slaqr5)(&wantT_,
                        &wantZ_,
                        &kacc22,
                        &n,
                        &kTop,
                        &kBot,
                        &nShifts,
                        sr,
                        si,
                        H,
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        Z,
                        &ldZ,
                        V,
                        &ldV,
                        U,
                        &ldU,
                        &nv,
                        WV,
                        &ldWV,
                        &nh,
                        WH,
                        &ldWH);
}


template <typename IndexType>
void
laqr5(bool          wantT,
      bool          wantZ,
      IndexType     kacc22,
      IndexType     n,
      IndexType     kTop,
      IndexType     kBot,
      IndexType     nShifts,
      double        *sr,
      double        *si,
      double        *H,
      IndexType     ldH,
      IndexType     iLoZ,
      IndexType     iHiZ,
      double        *Z,
      IndexType     ldZ,
      double        *V,
      IndexType     ldV,
      double        *U,
      IndexType     ldU,
      IndexType     nv,
      double        *WV,
      IndexType     ldWV,
      IndexType     nh,
      double        *WH,
      IndexType     ldWH)
{
    CXXLAPACK_DEBUG_OUT("dlaqr5");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(dlaqr5)(&wantT_,
                        &wantZ_,
                        &kacc22,
                        &n,
                        &kTop,
                        &kBot,
                        &nShifts,
                        sr,
                        si,
                        H,
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        Z,
                        &ldZ,
                        V,
                        &ldV,
                        U,
                        &ldU,
                        &nv,
                        WV,
                        &ldWV,
                        &nh,
                        WH,
                        &ldWH);
}

template <typename IndexType>
void
laqr5(bool                  wantT,
      bool                  wantZ,
      IndexType             kacc22,
      IndexType             n,
      IndexType             kTop,
      IndexType             kBot,
      IndexType             nShifts,
      std::complex<float >  *s,
      std::complex<float >  *H,
      IndexType             ldH,
      IndexType             iLoZ,
      IndexType             iHiZ,
      std::complex<float >  *Z,
      IndexType             ldZ,
      std::complex<float >  *V,
      IndexType             ldV,
      std::complex<float >  *U,
      IndexType             ldU,
      IndexType             nv,
      std::complex<float >  *WV,
      IndexType             ldWV,
      IndexType             nh,
      std::complex<float >  *WH,
      IndexType             ldWH)
{
    CXXLAPACK_DEBUG_OUT("claqr5");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(claqr5)(&wantT_,
                        &wantZ_,
                        &kacc22,
                        &n,
                        &kTop,
                        &kBot,
                        &nShifts,
                        reinterpret_cast<float  *>(s),
                        reinterpret_cast<float  *>(H),
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
                        reinterpret_cast<float  *>(V),
                        &ldV,
                        reinterpret_cast<float  *>(U),
                        &ldU,
                        &nv,
                        reinterpret_cast<float  *>(WV),
                        &ldWV,
                        &nh,
                        reinterpret_cast<float  *>(WH),
                        &ldWH);
}

template <typename IndexType>
void
laqr5(bool                  wantT,
      bool                  wantZ,
      IndexType             kacc22,
      IndexType             n,
      IndexType             kTop,
      IndexType             kBot,
      IndexType             nShifts,
      std::complex<double>  *s,
      std::complex<double>  *H,
      IndexType             ldH,
      IndexType             iLoZ,
      IndexType             iHiZ,
      std::complex<double>  *Z,
      IndexType             ldZ,
      std::complex<double>  *V,
      IndexType             ldV,
      std::complex<double>  *U,
      IndexType             ldU,
      IndexType             nv,
      std::complex<double>  *WV,
      IndexType             ldWV,
      IndexType             nh,
      std::complex<double>  *WH,
      IndexType             ldWH)
{
    CXXLAPACK_DEBUG_OUT("zlaqr5");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(zlaqr5)(&wantT_,
                        &wantZ_,
                        &kacc22,
                        &n,
                        &kTop,
                        &kBot,
                        &nShifts,
                        reinterpret_cast<double *>(s),
                        reinterpret_cast<double *>(H),
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
                        reinterpret_cast<double *>(V),
                        &ldV,
                        reinterpret_cast<double *>(U),
                        &ldU,
                        &nv,
                        reinterpret_cast<double *>(WV),
                        &ldWV,
                        &nh,
                        reinterpret_cast<double *>(WH),
                        &ldWH);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQR5_TCC

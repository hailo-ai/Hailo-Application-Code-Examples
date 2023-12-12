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

#ifndef CXXLAPACK_INTERFACE_LAQR3_TCC
#define CXXLAPACK_INTERFACE_LAQR3_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqr3(bool          wantT,
      bool          wantZ,
      IndexType     n,
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
      IndexType     lWork)
{
    CXXLAPACK_DEBUG_OUT("slaqr3");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(slaqr3)(&wantT_,
                        &wantZ_,
                        &n,
                        &kTop,
                        &kBot,
                        &nw,
                        H,
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        Z,
                        &ldZ,
                        &ns,
                        &nd,
                        sr,
                        si,
                        V,
                        &ldV,
                        &nh,
                        T,
                        &ldT,
                        &nv,
                        WV,
                        &ldWV,
                        work,
                        &lWork);
}


template <typename IndexType>
void
laqr3(bool          wantT,
      bool          wantZ,
      IndexType     n,
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
      IndexType     lWork)
{
    CXXLAPACK_DEBUG_OUT("dlaqr3");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(dlaqr3)(&wantT_,
                        &wantZ_,
                        &n,
                        &kTop,
                        &kBot,
                        &nw,
                        H,
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        Z,
                        &ldZ,
                        &ns,
                        &nd,
                        sr,
                        si,
                        V,
                        &ldV,
                        &nh,
                        T,
                        &ldT,
                        &nv,
                        WV,
                        &ldWV,
                        work,
                        &lWork);
}

template <typename IndexType>
void
laqr3(bool                      wantT,
      bool                      wantZ,
      IndexType                 n,
      IndexType                 kTop,
      IndexType                 kBot,
      IndexType                 nw,
      std::complex<float >      *H,
      IndexType                 ldH,
      IndexType                 iLoZ,
      IndexType                 iHiZ,
      std::complex<float >      *Z,
      IndexType                 ldZ,
      IndexType                 &ns,
      IndexType                 &nd,
      std::complex<float >      *sh,
      std::complex<float >      *V,
      IndexType                 ldV,
      IndexType                 nh,
      std::complex<float >      *T,
      IndexType                 ldT,
      IndexType                 nv,
      std::complex<float >      *WV,
      IndexType                 ldWV,
      std::complex<float >      *work,
      IndexType                 lWork)
{
    CXXLAPACK_DEBUG_OUT("claqr3");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(claqr3)(&wantT_,
                        &wantZ_,
                        &n,
                        &kTop,
                        &kBot,
                        &nw,
                        reinterpret_cast<float  *>(H),
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        reinterpret_cast<float  *>(Z),
                        &ldZ,
                        &ns,
                        &nd,
                        reinterpret_cast<float  *>(sh),
                        reinterpret_cast<float  *>(V),
                        &ldV,
                        &nh,
                        reinterpret_cast<float  *>(T),
                        &ldT,
                        &nv,
                        reinterpret_cast<float  *>(WV),
                        &ldWV,
                        reinterpret_cast<float  *>(work),
                        &lWork);
}

template <typename IndexType>
void
laqr3(bool                      wantT,
      bool                      wantZ,
      IndexType                 n,
      IndexType                 kTop,
      IndexType                 kBot,
      IndexType                 nw,
      std::complex<double>      *H,
      IndexType                 ldH,
      IndexType                 iLoZ,
      IndexType                 iHiZ,
      std::complex<double>      *Z,
      IndexType                 ldZ,
      IndexType                 &ns,
      IndexType                 &nd,
      std::complex<double>      *sh,
      std::complex<double>      *V,
      IndexType                 ldV,
      IndexType                 nh,
      std::complex<double>      *T,
      IndexType                 ldT,
      IndexType                 nv,
      std::complex<double>      *WV,
      IndexType                 ldWV,
      std::complex<double>      *work,
      IndexType                 lWork)
{
    CXXLAPACK_DEBUG_OUT("zlaqr3");

    IndexType wantT_ = wantT;
    IndexType wantZ_ = wantZ;
    LAPACK_IMPL(zlaqr3)(&wantT_,
                        &wantZ_,
                        &n,
                        &kTop,
                        &kBot,
                        &nw,
                        reinterpret_cast<double *>(H),
                        &ldH,
                        &iLoZ,
                        &iHiZ,
                        reinterpret_cast<double *>(Z),
                        &ldZ,
                        &ns,
                        &nd,
                        reinterpret_cast<double *>(sh),
                        reinterpret_cast<double *>(V),
                        &ldV,
                        &nh,
                        reinterpret_cast<double *>(T),
                        &ldT,
                        &nv,
                        reinterpret_cast<double *>(WV),
                        &ldWV,
                        reinterpret_cast<double *>(work),
                        &lWork);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQR3_TCC

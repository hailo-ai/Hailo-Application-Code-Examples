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

#ifndef CXXLAPACK_INTERFACE_LAR1V_TCC
#define CXXLAPACK_INTERFACE_LAR1V_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
lar1v(IndexType             n,
      IndexType             b1,
      IndexType             bn,
      float                 lambda,
      const float           *d,
      const float           *l,
      const float           *ld,
      const float           *lld,
      float                 pivmin,
      float                 gaptol,
      float                 *z,
      bool                  wantnc,
      IndexType             &negcnt,
      float                 &ztz,
      float                 &mingma,
      IndexType             &r,
      IndexType             *isuppz,
      float                 &nrminv,
      float                 &resid,
      float                 &rqcorr,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("slaq1v");

    bool wantnc_ = wantnc;
    LAPACK_IMPL(slar1v)(&n,
                        &b1,
                        &bn,
                        &lambda,
                        d,
                        l,
                        ld,
                        lld,
                        &pivmin,
                        &gaptol,
                        z,
                        &wantnc_,
                        &negcnt,
                        &ztz,
                        &mingma,
                        &r,
                        isuppz,
                        &nrminv,
                        &resid,
                        &rqcorr,
                        work);
}


template <typename IndexType>
void
lar1v(IndexType             n,
      IndexType             b1,
      IndexType             bn,
      double                lambda,
      const double          *d,
      const double          *l,
      const double          *ld,
      const double          *lld,
      double                pivmin,
      double                gaptol,
      double                *z,
      bool                  wantnc,
      IndexType             &negcnt,
      double                &ztz,
      double                &mingma,
      IndexType             &r,
      IndexType             *isuppz,
      double                &nrminv,
      double                &resid,
      double                &rqcorr,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dlaq1v");

    bool wantnc_ = wantnc;
    LAPACK_IMPL(dlar1v)(&n,
                        &b1,
                        &bn,
                        &lambda,
                        d,
                        l,
                        ld,
                        lld,
                        &pivmin,
                        &gaptol,
                        z,
                        &wantnc_,
                        &negcnt,
                        &ztz,
                        &mingma,
                        &r,
                        isuppz,
                        &nrminv,
                        &resid,
                        &rqcorr,
                        work);
}

template <typename IndexType>
void
lar1v(IndexType             n,
      IndexType             b1,
      IndexType             bn,
      float                 lambda,
      const float           *d,
      const float           *l,
      const float           *ld,
      const float           *lld,
      float                 pivmin,
      float                 gaptol,
      std::complex<float >  *z,
      bool                  wantnc,
      IndexType             &negcnt,
      float                 &ztz,
      float                 &mingma,
      IndexType             &r,
      IndexType             *isuppz,
      float                 &nrminv,
      float                 &resid,
      float                 &rqcorr,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("claq1v");

    bool wantnc_ = wantnc;
    LAPACK_IMPL(clar1v)(&n,
                        &b1,
                        &bn,
                        &lambda,
                        d,
                        l,
                        ld,
                        lld,
                        &pivmin,
                        &gaptol,
                        reinterpret_cast<float  *>(z),
                        &wantnc_,
                        &negcnt,
                        &ztz,
                        &mingma,
                        &r,
                        isuppz,
                        &nrminv,
                        &resid,
                        &rqcorr,
                        work);
}

template <typename IndexType>
void
lar1v(IndexType             n,
      IndexType             b1,
      IndexType             bn,
      double                lambda,
      const double          *d,
      const double          *l,
      const double          *ld,
      const double          *lld,
      double                pivmin,
      double                gaptol,
      std::complex<double>  *z,
      bool                  wantnc,
      IndexType             &negcnt,
      double                &ztz,
      double                &mingma,
      IndexType             &r,
      IndexType             *isuppz,
      double                &nrminv,
      double                &resid,
      double                &rqcorr,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("zlaq1v");

    bool wantnc_ = wantnc;
    LAPACK_IMPL(zlar1v)(&n,
                        &b1,
                        &bn,
                        &lambda,
                        d,
                        l,
                        ld,
                        lld,
                        &pivmin,
                        &gaptol,
                        reinterpret_cast<double *>(z),
                        &wantnc_,
                        &negcnt,
                        &ztz,
                        &mingma,
                        &r,
                        isuppz,
                        &nrminv,
                        &resid,
                        &rqcorr,
                        work);
}


} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAR1V_TCC

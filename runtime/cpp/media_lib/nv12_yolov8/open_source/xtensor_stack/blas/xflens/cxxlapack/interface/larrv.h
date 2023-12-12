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

#ifndef CXXLAPACK_INTERFACE_LARRV_H
#define CXXLAPACK_INTERFACE_LARRV_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    larrv(IndexType             n,
          float                 vl,
          float                 vu,
          float                 *d,
          float                 *l,
          float                 pivmin,
          IndexType             *isplit,
          float                 m,
          IndexType             dol,
          IndexType             dou,
          float                 minrgp,
          float                 rtol1,
          float                 rtol2,
          float                 *w,
          float                 *werr,
          float                 *wgap,
          const IndexType       *iblock,
          const IndexType       *indexw,
          const float           *gers,
          float                 *Z,
          IndexType             ldZ,
          IndexType             *isuppz,
          float                 *work,
          IndexType             *iWork);


template <typename IndexType>
    IndexType
    larrv(IndexType             n,
          double                vl,
          double                vu,
          double                *d,
          double                *l,
          double                pivmin,
          IndexType             *isplit,
          double                m,
          IndexType             dol,
          IndexType             dou,
          double                minrgp,
          double                rtol1,
          double                rtol2,
          double                *w,
          double                *werr,
          double                *wgap,
          const IndexType       *iblock,
          const IndexType       *indexw,
          const double          *gers,
          double                *Z,
          IndexType             ldZ,
          IndexType             *isuppz,
          double                *work,
          IndexType             *iWork);

template <typename IndexType>
    IndexType
    larrv(IndexType             n,
          float                 vl,
          float                 vu,
          float                 *d,
          float                 *l,
          float                 pivmin,
          IndexType             *isplit,
          float                 m,
          IndexType             dol,
          IndexType             dou,
          float                 minrgp,
          float                 rtol1,
          float                 rtol2,
          float                 *w,
          float                 *werr,
          float                 *wgap,
          const IndexType       *iblock,
          const IndexType       *indexw,
          const float           *gers,
          std::complex<float >  *Z,
          IndexType             ldZ,
          IndexType             *isuppz,
          float                 *work,
          IndexType             *iWork);

template <typename IndexType>
    IndexType
    larrv(IndexType             n,
          double                vl,
          double                vu,
          double                *d,
          double                *l,
          double                pivmin,
          IndexType             *isplit,
          double                m,
          IndexType             dol,
          IndexType             dou,
          double                minrgp,
          double                rtol1,
          double                rtol2,
          double                *w,
          double                *werr,
          double                *wgap,
          const IndexType       *iblock,
          const IndexType       *indexw,
          const double          *gers,
          std::complex<double>  *Z,
          IndexType             ldZ,
          IndexType             *isuppz,
          double                *work,
          IndexType             *iWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARRV_H

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

#ifndef CXXLAPACK_INTERFACE_LAED8_H
#define CXXLAPACK_INTERFACE_LAED8_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    laed8(IndexType             icomq,
          IndexType             &k,
          IndexType             n,
          IndexType             qsiz,
          float                 *d,
          float                 *Q,
          IndexType             ldQ,
          const IndexType       *indxq,
          float                 &rho,
          const IndexType       *cutpnt,
          const float           *z,
          float                 *dlambda,
          float                 *Q2,
          IndexType             ldQ2,
          float                 *w,
          IndexType             *perm,
          IndexType             &givptr,
          IndexType             &givcol,
          float                 *givnum,
          IndexType             *indxp,
          IndexType             *indx);

template <typename IndexType>
    IndexType
    laed8(IndexType             icomq,
          IndexType             &k,
          IndexType             n,
          IndexType             qsiz,
          double                *d,
          double                *Q,
          IndexType             ldQ,
          const IndexType       *indxq,
          double                &rho,
          const IndexType       *cutpnt,
          const double          *z,
          double                *dlambda,
          double                *Q2,
          IndexType             ldQ2,
          double                *w,
          IndexType             *perm,
          IndexType             &givptr,
          IndexType             &givcol,
          double                *givnum,
          IndexType             *indxp,
          IndexType             *indx);

template <typename IndexType>
    IndexType
    laed8(IndexType             &k,
          IndexType             n,
          IndexType             qsiz,
          std::complex<float >  *Q,
          IndexType             ldQ,
          float                 *d,
          float                 &rho,
          const IndexType       *cutpnt,
          const float           *z,
          float                 *dlambda,
          std::complex<float >  *Q2,
          IndexType             ldQ2,
          float                 *w,
          IndexType             *indxp,
          IndexType             *indx,
          const IndexType       *indxq,
          IndexType             *perm,
          IndexType             &givptr,
          IndexType             &givcol,
          float                 *givnum);

template <typename IndexType>
    IndexType
    laed8(IndexType             &k,
          IndexType             n,
          IndexType             qsiz,
          std::complex<double>  *Q,
          IndexType             ldQ,
          double                *d,
          double                &rho,
          const IndexType       *cutpnt,
          const double          *z,
          double                *dlambda,
          std::complex<double>  *Q2,
          IndexType             ldQ2,
          double                *w,
          IndexType             *indxp,
          IndexType             *indx,
          const IndexType       *indxq,
          IndexType             *perm,
          IndexType             &givptr,
          IndexType             &givcol,
          double                *givnum);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAED8_H

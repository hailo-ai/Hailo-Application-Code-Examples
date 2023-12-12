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

#ifndef CXXLAPACK_INTERFACE_LALSA_H
#define CXXLAPACK_INTERFACE_LALSA_H 1

#include <complex>

namespace cxxlapack {

template <typename IndexType>
    IndexType
    lalsa(IndexType             icompq,
          IndexType             smlsiz,
          IndexType             n,
          IndexType             nRhs,
          float                 *B,
          IndexType             ldB,
          float                 *Bx,
          IndexType             ldBx,
          const float           *U,
          IndexType             ldU,
          const float           *VT,
          const IndexType       *k,
          const float           *Difl,
          const float           *Difr,
          const float           *Z,
          const float           *Poles,
          const IndexType       *Givptr,
          const IndexType       *Givcol,
          IndexType             ldGcol,
          const IndexType       *Perm,
          const float           *Givnum,
          const float           *c,
          const float           *s,
          float                 *work,
          IndexType             *iWork);

template <typename IndexType>
    IndexType
    lalsa(IndexType             icompq,
          IndexType             smlsiz,
          IndexType             n,
          IndexType             nRhs,
          double                *B,
          IndexType             ldB,
          double                *Bx,
          IndexType             ldBx,
          const double          *U,
          IndexType             ldU,
          const double          *VT,
          const IndexType       *k,
          const double          *Difl,
          const double          *Difr,
          const double          *Z,
          const double          *Poles,
          const IndexType       *Givptr,
          const IndexType       *Givcol,
          IndexType             ldGcol,
          const IndexType       *Perm,
          const double          *Givnum,
          const double          *c,
          const double          *s,
          double                *work,
          IndexType             *iWork);

template <typename IndexType>
    IndexType
    lalsa(IndexType             icompq,
          IndexType             smlsiz,
          IndexType             n,
          IndexType             nRhs,
          std::complex<float >  *B,
          IndexType             ldB,
          std::complex<float >  *Bx,
          IndexType             ldBx,
          const float           *U,
          IndexType             ldU,
          const float           *VT,
          const IndexType       *k,
          const float           *Difl,
          const float           *Difr,
          const float           *Z,
          const float           *Poles,
          const IndexType       *Givptr,
          const IndexType       *Givcol,
          IndexType             ldGcol,
          const IndexType       *Perm,
          const float           *Givnum,
          const float           *c,
          const float           *s,
          float                 *work,
          IndexType             *iWork);

template <typename IndexType>
    IndexType
    lalsa(IndexType             icompq,
          IndexType             smlsiz,
          IndexType             n,
          IndexType             nRhs,
          std::complex<double>  *B,
          IndexType             ldB,
          std::complex<double>  *Bx,
          IndexType             ldBx,
          const double          *U,
          IndexType             ldU,
          const double          *VT,
          const IndexType       *k,
          const double          *Difl,
          const double          *Difr,
          const double          *Z,
          const double          *Poles,
          const IndexType       *Givptr,
          const IndexType       *Givcol,
          IndexType             ldGcol,
          const IndexType       *Perm,
          const double          *Givnum,
          const double          *c,
          const double          *s,
          double                *work,
          IndexType             *iWork);

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LALSA_H

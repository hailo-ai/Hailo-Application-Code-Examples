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

#ifndef CXXLAPACK_INTERFACE_LASD7_TCC
#define CXXLAPACK_INTERFACE_LASD7_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
lasd7(IndexType             icompq,
      IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             &k,
      float                 *d,
      float                 *z,
      float                 *zw,
      float                 *vf,
      float                 *vfw,
      float                 *vl,
      float                 *vlw,
      float                 &alpha,
      float                 &beta,
      float                 *dsigma,
      IndexType             *idx,
      IndexType             *idxp,
      const IndexType       *idxq,
      IndexType             *perm,
      IndexType             &givptr,
      IndexType             *Givcol,
      IndexType             ldGcol,
      float                 *Givnum,
      IndexType             ldGnum,
      float                 &c,
      float                 &s)
{
    CXXLAPACK_DEBUG_OUT("slasd7");

    IndexType info;
    LAPACK_IMPL(slasd7)(&icompq,
                        &nl,
                        &nr,
                        &sqre,
                        &k,
                        d,
                        z,
                        zw,
                        vf,
                        vfw,
                        vl,
                        vlw,
                        &alpha,
                        &beta,
                        dsigma,
                        idx,
                        idxp,
                        idxq,
                        perm,
                        &givptr,
                        Givcol,
                        &ldGcol,
                        Givnum,
                        &ldGnum,
                        &c,
                        &s,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
lasd7(IndexType             icompq,
      IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             &k,
      double                *d,
      double                *z,
      double                *zw,
      double                *vf,
      double                *vfw,
      double                *vl,
      double                *vlw,
      double                &alpha,
      double                &beta,
      double                *dsigma,
      IndexType             *idx,
      IndexType             *idxp,
      const IndexType       *idxq,
      IndexType             *perm,
      IndexType             &givptr,
      IndexType             *Givcol,
      IndexType             ldGcol,
      double                *Givnum,
      IndexType             ldGnum,
      double                &c,
      double                &s)
{
    CXXLAPACK_DEBUG_OUT("dlasd7");

    IndexType info;
    LAPACK_IMPL(dlasd7)(&icompq,
                        &nl,
                        &nr,
                        &sqre,
                        &k,
                        d,
                        z,
                        zw,
                        vf,
                        vfw,
                        vl,
                        vlw,
                        &alpha,
                        &beta,
                        dsigma,
                        idx,
                        idxp,
                        idxq,
                        perm,
                        &givptr,
                        Givcol,
                        &ldGcol,
                        Givnum,
                        &ldGnum,
                        &c,
                        &s,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LASD7_TCC

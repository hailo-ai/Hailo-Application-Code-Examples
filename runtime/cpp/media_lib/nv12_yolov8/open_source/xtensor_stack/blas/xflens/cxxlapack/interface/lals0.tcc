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

#ifndef CXXLAPACK_INTERFACE_LALS0_TCC
#define CXXLAPACK_INTERFACE_LALS0_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
lals0(IndexType             icomq,
      IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             nRhs,
      float                 *B,
      IndexType             ldB,
      float                 *Bx,
      IndexType             ldBx,
      const IndexType       *perm,
      IndexType             givptr,
      const IndexType       *Givcol,
      IndexType             ldGcol,
      const float           *Givnum,
      IndexType             ldGnum,
      const float           *Poles,
      const float           *difl,
      const float           *difr,
      const float           *z,
      IndexType             k,
      float                 c,
      float                 s,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("slals0");

    IndexType info;
    LAPACK_IMPL(slals0)(&icomq,
                        &nl,
                        &nr,
                        &sqre,
                        &nRhs,
                        B,
                        &ldB,
                        Bx,
                        &ldBx,
                        perm,
                        givptr,
                        Givcol,
                        &ldGcol,
                        Givnum,
                        &ldGnum,
                        Poles,
                        difl,
                        difr,
                        z,
                        &k,
                        &c,
                        &s,
                        work,
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
lals0(IndexType             icomq,
      IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             nRhs,
      double                *B,
      IndexType             ldB,
      double                *Bx,
      IndexType             ldBx,
      const IndexType       *perm,
      IndexType             givptr,
      const IndexType       *Givcol,
      IndexType             ldGcol,
      const double          *Givnum,
      IndexType             ldGnum,
      const double          *Poles,
      const double          *difl,
      const double          *difr,
      const double          *z,
      IndexType             k,
      double                c,
      double                s,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dlals0");

    IndexType info;
    LAPACK_IMPL(dlals0)(&icomq,
                        &nl,
                        &nr,
                        &sqre,
                        &nRhs,
                        B,
                        &ldB,
                        Bx,
                        &ldBx,
                        perm,
                        givptr,
                        Givcol,
                        &ldGcol,
                        Givnum,
                        &ldGnum,
                        Poles,
                        difl,
                        difr,
                        z,
                        &k,
                        &c,
                        &s,
                        work,
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
lals0(IndexType             icomq,
      IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             nRhs,
      std::complex<float >  *B,
      IndexType             ldB,
      std::complex<float >  *Bx,
      IndexType             ldBx,
      const IndexType       *perm,
      IndexType             givptr,
      const IndexType       *Givcol,
      IndexType             ldGcol,
      const float           *Givnum,
      IndexType             ldGnum,
      const float           *Poles,
      const float           *difl,
      const float           *difr,
      const float           *z,
      IndexType             k,
      float                 c,
      float                 s,
      float                 *rWork)
{
    CXXLAPACK_DEBUG_OUT("clals0");

    IndexType info;
    LAPACK_IMPL(clals0)(&icomq,
                        &nl,
                        &nr,
                        &sqre,
                        &nRhs,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(Bx),
                        &ldBx,
                        perm,
                        givptr,
                        Givcol,
                        &ldGcol,
                        Givnum,
                        &ldGnum,
                        Poles,
                        difl,
                        difr,
                        z,
                        &k,
                        &c,
                        &s,
                        rWork,
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
lals0(IndexType             icomq,
      IndexType             nl,
      IndexType             nr,
      IndexType             sqre,
      IndexType             nRhs,
      std::complex<double>  *B,
      IndexType             ldB,
      std::complex<double>  *Bx,
      IndexType             ldBx,
      const IndexType       *perm,
      IndexType             givptr,
      const IndexType       *Givcol,
      IndexType             ldGcol,
      const double          *Givnum,
      IndexType             ldGnum,
      const double          *Poles,
      const double          *difl,
      const double          *difr,
      const double          *z,
      IndexType             k,
      double                c,
      double                s,
      double                *rWork)
{
    CXXLAPACK_DEBUG_OUT("zlals0");

    IndexType info;
    LAPACK_IMPL(zlals0)(&icomq,
                        &nl,
                        &nr,
                        &sqre,
                        &nRhs,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(Bx),
                        &ldBx,
                        perm,
                        givptr,
                        Givcol,
                        &ldGcol,
                        Givnum,
                        &ldGnum,
                        Poles,
                        difl,
                        difr,
                        z,
                        &k,
                        &c,
                        &s,
                        rWork,
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

#endif // CXXLAPACK_INTERFACE_LALS0_TCC

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

#ifndef CXXLAPACK_INTERFACE_LAQP2_TCC
#define CXXLAPACK_INTERFACE_LAQP2_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqp2(IndexType     m,
      IndexType     n,
      IndexType     offset,
      float         *A,
      IndexType     ldA,
      IndexType     *jPvt,
      float         *tau,
      float         *vn1,
      float         *vn2,
      float         *work)
{
    CXXLAPACK_DEBUG_OUT("slaqp2");

    LAPACK_IMPL(slaqp2)(&m,
                        &n,
                        &offset,
                        A,
                        &ldA,
                        jPvt,
                        tau,
                        vn1,
                        vn2,
                        work);

}


template <typename IndexType>
void
laqp2(IndexType     m,
      IndexType     n,
      IndexType     offset,
      double        *A,
      IndexType     ldA,
      IndexType     *jPvt,
      double        *tau,
      double        *vn1,
      double        *vn2,
      double        *work)
{
    CXXLAPACK_DEBUG_OUT("dlaqp2");

    LAPACK_IMPL(dlaqp2)(&m,
                        &n,
                        &offset,
                        A,
                        &ldA,
                        jPvt,
                        tau,
                        vn1,
                        vn2,
                        work);

}

template <typename IndexType>
void
laqp2(IndexType             m,
      IndexType             n,
      IndexType             offset,
      std::complex<float >  *A,
      IndexType             ldA,
      IndexType             *jPvt,
      std::complex<float >  *tau,
      float                 *vn1,
      float                 *vn2,
      std::complex<float >  *work)
{
    CXXLAPACK_DEBUG_OUT("claqp2");

    LAPACK_IMPL(claqp2)(&m,
                        &n,
                        &offset,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        jPvt,
                        reinterpret_cast<float  *>(tau),
                        vn1,
                        vn2,
                        reinterpret_cast<float  *>(work));
}

template <typename IndexType>
void
laqp2(IndexType             m,
      IndexType             n,
      IndexType             offset,
      std::complex<double>  *A,
      IndexType             ldA,
      IndexType             *jPvt,
      std::complex<double>  *tau,
      double                *vn1,
      double                *vn2,
      std::complex<double>  *work)
{
    CXXLAPACK_DEBUG_OUT("zlaqp2");

    LAPACK_IMPL(zlaqp2)(&m,
                        &n,
                        &offset,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        jPvt,
                        reinterpret_cast<double *>(tau),
                        vn1,
                        vn2,
                        reinterpret_cast<double *>(work));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQP2_TCC

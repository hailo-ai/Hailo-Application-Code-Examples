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

#ifndef CXXLAPACK_INTERFACE_LASQ6_TCC
#define CXXLAPACK_INTERFACE_LASQ6_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
lasq6(IndexType             i0,
      IndexType             n0,
      const float           *z,
      IndexType             pp,
      float                 &dmin,
      float                 &dmin1,
      float                 &dmin2,
      float                 &dn,
      float                 &dnm1,
      float                 &dnm2)
{
    CXXLAPACK_DEBUG_OUT("slasq6");

    IndexType info;
    LAPACK_IMPL(slasq6)(&i0,
                        &n0,
                        z,
                        &pp,
                        &dmin,
                        &dmin1,
                        &dmin2,
                        &dn,
                        &dnm1,
                        &dnm2);

}

template <typename IndexType>
void
lasq6(IndexType             i0,
      IndexType             n0,
      const double          *z,
      IndexType             pp,
      double                &dmin,
      double                &dmin1,
      double                &dmin2,
      double                &dn,
      double                &dnm1,
      double                &dnm2)
{
    CXXLAPACK_DEBUG_OUT("dlasq6");

    IndexType info;
    LAPACK_IMPL(dlasq6)(&i0,
                        &n0,
                        z,
                        &pp,
                        &dmin,
                        &dmin1,
                        &dmin2,
                        &dn,
                        &dnm1,
                        &dnm2);

}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LASQ6_TCC

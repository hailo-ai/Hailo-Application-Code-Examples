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

#ifndef CXXLAPACK_INTERFACE_LANV2_TCC
#define CXXLAPACK_INTERFACE_LANV2_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename XFLENS_VOID>
void
lanv2(float    &a,
      float    &b,
      float    &c,
      float    &d,
      float    &rt1r,
      float    &rt1i,
      float    &rt2r,
      float    &rt2i,
      float    &cs,
      float    &sn)
{
    CXXLAPACK_DEBUG_OUT("slanv2");

    LAPACK_IMPL(slanv2)(&a,
                        &b,
                        &c,
                        &d,
                        &rt1r,
                        &rt1i,
                        &rt2r,
                        &rt2i,
                        &cs,
                        &sn);
}

template <typename XFLENS_VOID>
void
lanv2(double   &a,
      double   &b,
      double   &c,
      double   &d,
      double   &rt1r,
      double   &rt1i,
      double   &rt2r,
      double   &rt2i,
      double   &cs,
      double   &sn)
{

    CXXLAPACK_DEBUG_OUT("dlanv2");

    LAPACK_IMPL(dlanv2)(&a,
                        &b,
                        &c,
                        &d,
                        &rt1r,
                        &rt1i,
                        &rt2r,
                        &rt2i,
                        &cs,
                        &sn);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LANV2_TCC

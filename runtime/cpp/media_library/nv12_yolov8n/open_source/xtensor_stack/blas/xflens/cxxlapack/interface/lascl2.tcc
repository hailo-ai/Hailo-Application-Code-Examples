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

#ifndef CXXLAPACK_INTERFACE_LASCL2_TCC
#define CXXLAPACK_INTERFACE_LASCL2_TCC

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
lascl2(IndexType             m,
       IndexType             n,
       const float           *d,
       float                 *X,
       IndexType             ldX)
{
    CXXLAPACK_DEBUG_OUT("slascl2");

    LAPACK_IMPL(slascl2)(&m,
                         &n,
                         d,
                         X,
                         &ldX);
}


template <typename IndexType>
void
lascl2(IndexType             m,
       IndexType             n,
       const double          *d,
       double                *X,
       IndexType             ldX)
{
    CXXLAPACK_DEBUG_OUT("dlascl2");

    LAPACK_IMPL(dlascl2)(&m,
                         &n,
                         d,
                         X,
                         &ldX);
}


template <typename IndexType>
void
lascl2(IndexType             m,
       IndexType             n,
       const float           *d,
       std::complex<float >  *X,
       IndexType             ldX)
{
    CXXLAPACK_DEBUG_OUT("clascl2");

    LAPACK_IMPL(clascl2)(&m,
                         &n,
                         d,
                         reinterpret_cast<float  *>(X),
                         &ldX);
}

template <typename IndexType>
void
lascl2(IndexType             m,
       IndexType             n,
       const double          *d,
       std::complex<double>  *X,
       IndexType             ldX)
{
    CXXLAPACK_DEBUG_OUT("zlascl2");

    LAPACK_IMPL(zlascl2)(&m,
                         &n,
                         d,
                         reinterpret_cast<double *>(X),
                         &ldX);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LASCL2_TCC

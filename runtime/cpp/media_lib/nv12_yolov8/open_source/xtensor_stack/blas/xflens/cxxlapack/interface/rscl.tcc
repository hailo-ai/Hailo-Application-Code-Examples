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

#ifndef CXXLAPACK_INTERFACE_RSCL_TCC
#define CXXLAPACK_INTERFACE_RSCL_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
rscl(IndexType          n,
     const float        &sa,
     float              *sx,
     IndexType          incX)
{
    CXXLAPACK_DEBUG_OUT("srscl");

    LAPACK_IMPL(srscl)(&n,
                       &sa,
                       sx,
                       &incX);
}

template <typename IndexType>
void
rscl(IndexType          n,
     const double       &sa,
     double             *sx,
     IndexType          incX)
{
    CXXLAPACK_DEBUG_OUT("drscl");

    LAPACK_IMPL(drscl)(&n,
                       &sa,
                       sx,
                       &incX);
}

template <typename IndexType>
void
rscl(IndexType              n,
     const float            &sa,
     std::complex<float >   *sx,
     IndexType              incX)
{
    CXXLAPACK_DEBUG_OUT("csrscl");

    LAPACK_IMPL(csrscl)(&n,
                        &sa,
                        reinterpret_cast<float  *>(sx),
                        &incX);
}

template <typename IndexType>
void
rscl(IndexType              n,
     const double           &sa,
     std::complex<double>   *sx,
     IndexType              incX)
{
    CXXLAPACK_DEBUG_OUT("zdrscl");

    LAPACK_IMPL(zdrscl)(&n,
                        &sa,
                        reinterpret_cast<double *>(sx),
                        &incX);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_RSCL_TCC

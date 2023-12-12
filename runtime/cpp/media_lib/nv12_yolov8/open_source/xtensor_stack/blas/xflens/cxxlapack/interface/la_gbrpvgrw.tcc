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

#ifndef CXXLAPACK_INTERFACE_LA_GBRPVGRW_TCC
#define CXXLAPACK_INTERFACE_LA_GBRPVGRW_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
double
la_gbrpvgrw(IndexType             n,
            IndexType             kl,
            IndexType             ku,
            IndexType             ncols,
            const float           *Ab,
            IndexType             ldAb,
            const float           *Afb,
            IndexType             ldAfb)
{
    CXXLAPACK_DEBUG_OUT("sla_gbrpvgrw");

    return LAPACK_IMPL(sla_gbrpvgrw)(&n,
                                     &kl,
                                     &ku,
                                     &ncols,
                                     Ab,
                                     &ldAb,
                                     Afb,
                                     &ldAfb);
}

template <typename IndexType>
double
la_gbrpvgrw(IndexType             n,
            IndexType             kl,
            IndexType             ku,
            IndexType             ncols,
            const double          *Ab,
            IndexType             ldAb,
            const double          *Afb,
            IndexType             ldAfb)
{
    CXXLAPACK_DEBUG_OUT("dla_gbrpvgrw");

    return LAPACK_IMPL(dla_gbrpvgrw)(&n,
                                     &kl,
                                     &ku,
                                     &ncols,
                                     Ab,
                                     &ldAb,
                                     Afb,
                                     &ldAfb);
}

template <typename IndexType>
double
la_gbrpvgrw(IndexType                     n,
            IndexType                     kl,
            IndexType                     ku,
            IndexType                     ncols,
            const std::complex<float >    *Ab,
            IndexType                     ldAb,
            const std::complex<float >    *Afb,
            IndexType                     ldAfb)
{
    CXXLAPACK_DEBUG_OUT("cla_gbrpvgrw");

    return LAPACK_IMPL(cla_gbrpvgrw)(&n,
                                     &kl,
                                     &ku,
                                     &ncols,
                                     reinterpret_cast<const float  *>(Ab),
                                     &ldAb,
                                     reinterpret_cast<const float  *>(Afb),
                                     &ldAfb);
}

template <typename IndexType>
double
la_gbrpvgrw(IndexType                     n,
            IndexType                     kl,
            IndexType                     ku,
            IndexType                     ncols,
            const std::complex<double>    *Ab,
            IndexType                     ldAb,
            const std::complex<double>    *Afb,
            IndexType                     ldAfb)
{
    CXXLAPACK_DEBUG_OUT("zla_gbrpvgrw");

    return LAPACK_IMPL(zla_gbrpvgrw)(&n,
                                     &kl,
                                     &ku,
                                     &ncols,
                                     reinterpret_cast<const double *>(Ab),
                                     &ldAb,
                                     reinterpret_cast<const double *>(Afb),
                                     &ldAfb);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LA_GBRPVGRW_TCC

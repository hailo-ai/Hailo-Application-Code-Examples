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

#ifndef CXXLAPACK_INTERFACE_LA_LIN_BERR_TCC
#define CXXLAPACK_INTERFACE_LA_LIN_BERR_TCC

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
la_lin_berr(IndexType             n,
            IndexType             nz,
            IndexType             nRhs,
            const float           *res,
            const float           *ayb,
            float                 *berr)
{
    CXXLAPACK_DEBUG_OUT("sla_lin_berr");

    LAPACK_IMPL(sla_lin_berr)(&n,
                              &nz,
                              &nRhs,
                              res,
                              ayb,
                              berr);
}

template <typename IndexType>
void
la_lin_berr(IndexType             n,
            IndexType             nz,
            IndexType             nRhs,
            const double          *res,
            const double          *ayb,
            double                *berr)
{
    CXXLAPACK_DEBUG_OUT("dla_lin_berr");

    LAPACK_IMPL(dla_lin_berr)(&n,
                              &nz,
                              &nRhs,
                              res,
                              ayb,
                              berr);
}

template <typename IndexType>
void
la_lin_berr(IndexType                   n,
            IndexType                   nz,
            IndexType                   nRhs,
            const std::complex<float >  *res,
            const float                 *ayb,
            float                       *berr)
{
    CXXLAPACK_DEBUG_OUT("cla_lin_berr");

    LAPACK_IMPL(cla_lin_berr)(&n,
                              &nz,
                              &nRhs,
                              reinterpret_cast<const float  *>(res),
                              ayb,
                              berr);
}

template <typename IndexType>
void
la_lin_berr(IndexType                   n,
            IndexType                   nz,
            IndexType                   nRhs,
            const std::complex<double>  *res,
            const double                *ayb,
            double                      *berr)
{
    CXXLAPACK_DEBUG_OUT("zla_lin_berr");

    LAPACK_IMPL(zla_lin_berr)(&n,
                              &nz,
                              &nRhs,
                              reinterpret_cast<const double *>(res),
                              ayb,
                              berr);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LA_LIN_BERR_TCC

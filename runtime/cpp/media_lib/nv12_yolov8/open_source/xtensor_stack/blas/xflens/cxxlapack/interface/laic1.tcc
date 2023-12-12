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

#ifndef CXXLAPACK_INTERFACE_LAIC1_TCC
#define CXXLAPACK_INTERFACE_LAIC1_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laic1(IndexType     job,
      IndexType     j,
      const float   *x,
      float         sEst,
      const float   *w,
      float         gamma,
      float         &sEstPr,
      float         &s,
      float         &c)
{
    CXXLAPACK_DEBUG_OUT("slaic1");

    LAPACK_IMPL(slaic1)(&job,
                        &j,
                        x,
                        &sEst,
                        w,
                        &gamma,
                        &sEstPr,
                        &s,
                        &c);
}


template <typename IndexType>
void
laic1(IndexType     job,
      IndexType     j,
      const double  *x,
      double        sEst,
      const double  *w,
      double        gamma,
      double        &sEstPr,
      double        &s,
      double        &c)
{
    CXXLAPACK_DEBUG_OUT("dlaic1");

    LAPACK_IMPL(dlaic1)(&job,
                        &j,
                        x,
                        &sEst,
                        w,
                        &gamma,
                        &sEstPr,
                        &s,
                        &c);
}

template <typename IndexType>
void
laic1(IndexType                   job,
      IndexType                   j,
      const std::complex<float >  *x,
      float                       sEst,
      const std::complex<float >  *w,
      const std::complex<float >  &gamma,
      float                       &sEstPr,
      std::complex<float >        &s,
      std::complex<float >        &c)
{
    CXXLAPACK_DEBUG_OUT("claic1");

    LAPACK_IMPL(claic1)(&job,
                        &j,
                        reinterpret_cast<const float  *>(x),
                        &sEst,
                        reinterpret_cast<const float  *>(w),
                        reinterpret_cast<const float  *>(&gamma),
                        &sEstPr,
                        reinterpret_cast<float  *>(&s),
                        reinterpret_cast<float  *>(&c));
}

template <typename IndexType>
void
laic1(IndexType                   job,
      IndexType                   j,
      const std::complex<double>  *x,
      double                      sEst,
      const std::complex<double>  *w,
      const std::complex<double>  &gamma,
      double                      &sEstPr,
      std::complex<double>        &s,
      std::complex<double>        &c)
{
    CXXLAPACK_DEBUG_OUT("zlaic1");

    LAPACK_IMPL(zlaic1)(&job,
                        &j,
                        reinterpret_cast<const double *>(x),
                        &sEst,
                        reinterpret_cast<const double *>(w),
                        reinterpret_cast<const double *>(&gamma),
                        &sEstPr,
                        reinterpret_cast<double *>(&s),
                        reinterpret_cast<double *>(&c));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAIC1_TCC

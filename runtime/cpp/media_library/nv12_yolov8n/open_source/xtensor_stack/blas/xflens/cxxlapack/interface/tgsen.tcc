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

#ifndef CXXLAPACK_INTERFACE_TGSEN_TCC
#define CXXLAPACK_INTERFACE_TGSEN_TCC 1

#include <iostream>
#include "xflens/cxxlapack/interface/interface.h"
#include "xflens/cxxlapack/netlib/netlib.h"

namespace cxxlapack {

/*template <typename IndexType>
IndexType
tgsen(IndexType             ijob,
       bool                  wantq,
       bool                  wantz,
       bool                  *select,
       IndexType             n,
       float                 *A,
       IndexType             ldA,
       float                 *B,
       IndexType             ldB,
       float                 *alphar,
       float                 *alphai,
       float                 *beta,
       float                 *Q,
       IndexType             ldQ,
       float                 *Z,
       IndexType             ldZ,
       IndexType             &m,
       float                 &pl,
       float                 &pr,
       float                 *dif,
       float                 *work,
       IndexType             lWork,
       IndexType             *iWork,
       IndexType             liWork)
{
    CXXLAPACK_DEBUG_OUT("stgsen");

     IndexType info;
     IndexType wantq_ = wantq;
     IndexType wantz_ = wantz;
     //TODO: Convert select into a logical array!
     LAPACK_IMPL(stgsen)(&ijob,
                         &wantq_,
                         &wantz_,
                         select,
                         &n,
                         A,
                         &ldA,
                         B,
                         &ldB,
                         alphar,
                         alphai,
                         beta,
                         Q,
                         &ldQ,
                         Z,
                         &ldZ,
                         &m,
                         &pl,
                         &pr,
                         dif,
                         work,
                         &lWork,
                         iWork,
                         &liWork,
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
tgsen(IndexType             ijob,
       bool                  wantq,
       bool                  wantz,
       bool                  *select,
       IndexType             n,
       double                *A,
       IndexType             ldA,
       double                *B,
       IndexType             ldB,
       double                *alphar,
       double                *alphai,
       double                *beta,
       double                *Q,
       IndexType             ldQ,
       double                *Z,
       IndexType             ldZ,
       IndexType             &m,
       double                &pl,
       double                &pr,
       double                *dif,
       double                *work,
       IndexType             lWork,
       IndexType             *iWork,
       IndexType             liWork)
{
     CXXLAPACK_DEBUG_OUT("dtgsen");

     IndexType info;
     IndexType wantq_ = wantq;
     IndexType wantz_ = wantz;
     //TODO: Convert select into a logical array!
     LAPACK_IMPL(dtgsen)(&ijob,
                         &wantq_,
                         &wantz_,
                         select,
                         &n,
                         A,
                         &ldA,
                         B,
                         &ldB,
                         alphar,
                         alphai,
                         beta,
                         Q,
                         &ldQ,
                         Z,
                         &ldZ,
                         &m,
                         &pl,
                         &pr,
                         dif,
                         work,
                         &lWork,
                         iWork,
                         &liWork,
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
tgsen(IndexType             ijob,
       bool                  wantq,
       bool                  wantz,
       bool                  *select,
       IndexType             n,
       std::complex<float >  *A,
       IndexType             ldA,
       std::complex<float >  *B,
       IndexType             ldB,
       std::complex<float >  *alpha,
       std::complex<float >  *beta,
       std::complex<float >  *Q,
       IndexType             ldQ,
       std::complex<float >  *Z,
       IndexType             ldZ,
       IndexType             &m,
       double                &pl,
       double                &pr,
       double                *dif,
       std::complex<float >  *work,
       IndexType             lWork,
       IndexType             *iWork,
       IndexType             liWork)
{
     CXXLAPACK_DEBUG_OUT("ctgsen");

     IndexType info;
     IndexType wantq_ = wantq;
     IndexType wantz_ = wantz;
     //TODO: Convert select into a logical array!
     LAPACK_IMPL(ctgsen)(&ijob,
                         &wantq_,
                         &wantz_,
                         select,
                         &n,
                         reinterpret_cast<float  *>(A),
                         &ldA,
                         reinterpret_cast<float  *>(B),
                         &ldB,
                         reinterpret_cast<float  *>(alpha),
                         reinterpret_cast<float  *>(beta),
                         reinterpret_cast<float  *>(Q),
                         &ldQ,
                         reinterpret_cast<float  *>(Z),
                         &ldZ,
                         &m,
                         &pl,
                         &pr,
                         dif,
                         reinterpret_cast<float  *>(work),
                         &lWork,
                         iWork,
                         &liWork,
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
tgsen(IndexType             ijob,
       bool                  wantq,
       bool                  wantz,
       bool                  *select,
       IndexType             n,
       std::complex<double>  *A,
       IndexType             ldA,
       std::complex<double>  *B,
       IndexType             ldB,
       std::complex<double>  *alpha,
       std::complex<double>  *beta,
       std::complex<double>  *Q,
       IndexType             ldQ,
       std::complex<double>  *Z,
       IndexType             ldZ,
       IndexType             &m,
       double                &pl,
       double                &pr,
       double                *dif,
       std::complex<double>  *work,
       IndexType             lWork,
       IndexType             *iWork,
       IndexType             liWork)
{
     CXXLAPACK_DEBUG_OUT("ztgsen");

     IndexType info;
     IndexType wantq_ = wantq;
     IndexType wantz_ = wantz;
     //TODO: Convert select into a logical array!
     LAPACK_IMPL(ztgsen)(&ijob,
                         &wantq_,
                         &wantz_,
                         select,
                         &n,
                         reinterpret_cast<double *>(A),
                         &ldA,
                         reinterpret_cast<double *>(B),
                         &ldB,
                         reinterpret_cast<double *>(alpha),
                         reinterpret_cast<double *>(beta),
                         reinterpret_cast<double *>(Q),
                         &ldQ,
                         reinterpret_cast<double *>(Z),
                         &ldZ,
                         &m,
                         &pl,
                         &pr,
                         dif,
                         reinterpret_cast<double *>(work),
                         &lWork,
                         iWork,
                         &liWork,
                         &info);
 #   ifndef NDEBUG
     if (info<0) {
         std::cerr << "info = " << info << std::endl;
     }
 #   endif
     ASSERT(info>=0);
     return info;
}*/

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_TGSEN_TCC

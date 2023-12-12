/*
 *   Copyright (c) 2009, Michael Lehn
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

#ifndef CXXBLAS_LEVEL2_SYR2_H
#define CXXBLAS_LEVEL2_SYR2_H 1

#include "xflens/cxxblas/drivers/drivers.h"
#include "xflens/cxxblas/typedefs.h"

#define HAVE_CXXBLAS_SYR2 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename VX, typename VY,
          typename MA>
    void
    syr2(StorageOrder order,  StorageUpLo upLo,
        IndexType n,
        const ALPHA &alpha,
        const VX *x, IndexType incX,
        const VY *y, IndexType incY,
        MA *A, IndexType ldA);

#ifdef HAVE_CBLAS
// ssyr2
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    syr2(StorageOrder order,   StorageUpLo upLo,
         IndexType n,
         float alpha,
         const float *x, IndexType incX,
         const float *y, IndexType incY,
         float *A, IndexType ldA);

// dsyr2
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    syr2(StorageOrder order,   StorageUpLo upLo,
         IndexType n,
         double alpha,
         const double *x, IndexType incX,
         const double *y, IndexType incY,
         double *A, IndexType ldA);

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_SYR2_H

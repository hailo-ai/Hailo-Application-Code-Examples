/*
 *   Copyright (c) 2010, Michael Lehn
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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_HESCAL_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_HESCAL_TCC 1

#include <cassert>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA>
void
hescal(StorageOrder order, StorageUpLo upLo,
       IndexType n,
       const ALPHA &alpha, MA *A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("hescal_generic");

    if (alpha==ALPHA(1)) {
        return;
    }
    if (order==ColMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
    }
    if (upLo==Upper) {
        for (IndexType i=0; i<n; ++i) {
            scal(n-i, alpha, A+i*(ldA+1), IndexType(1));
        }
    } else {
        for (IndexType i=0; i<n; ++i) {
            scal(i+1, alpha, A+i*ldA, IndexType(1));
        }
    }
    for (IndexType i=0; i<n; ++i) {
        A[i*(ldA+1)] = cxxblas::real(A[i*(ldA+1)]);
    }
}

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_HESCAL_TCC

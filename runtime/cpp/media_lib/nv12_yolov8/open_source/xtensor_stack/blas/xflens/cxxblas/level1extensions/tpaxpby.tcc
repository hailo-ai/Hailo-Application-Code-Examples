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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_TPAXPBY_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_TPAXPBY_TCC 1

#include <algorithm>
#include <cassert>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

//
//  B = beta*B + alpha*op(A)
//
template <typename IndexType, typename ALPHA, typename MA,
          typename BETA, typename MB>
void
tpaxpby(StorageOrder order, StorageUpLo upLo, Transpose trans, Diag diag,
        IndexType n, const ALPHA &alpha, const MA *A, const BETA &beta, MB *B)
{
    CXXBLAS_DEBUG_OUT("tpaxpby_generic");

    FAKE_USE(order);
    FAKE_USE(upLo);

    const IndexType shift = (diag==Unit) ? 1 : 0;

    // TODO: Remove copy of diagonal if diag == Unit

    if (trans==NoTrans) {
          const IndexType length = n*(n+1)/2;
          axpby(length, alpha, A, 1, beta, B, 1);
    } else if (trans==Conj) {
          const IndexType length = n*(n+1)/2;
          acxpby(length, alpha, A, 1, beta, B, 1);
    } else if (trans == Trans) {
        for (IndexType i = 0; i < n; ++i) {
            for (IndexType j = i+shift; j < n; ++j) {
                B[i+j*(j+1)/2] = beta*B[i+j*(j+1)/2] + alpha*A[j+(2*n-i-1)*i/2];
            }
        }
    } else if ( trans==ConjTrans) {
        for (IndexType i = 0; i < n; ++i) {
            for (IndexType j = i+shift; j < n; ++j) {
                B[i+j*(j+1)/2] = beta*B[i+j*(j+1)/2]
                               + alpha*conjugate(A[j+(2*n-i-1)*i/ 2]);
            }
        }
    }
    return;
}

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_TPAXPBY_TCC

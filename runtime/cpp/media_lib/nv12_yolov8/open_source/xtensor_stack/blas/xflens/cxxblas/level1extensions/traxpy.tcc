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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_TRAXPY_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_TRAXPY_TCC 1

#include <algorithm>
#include <cassert>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

//
//  B += alpha*op(A)
//
template <typename IndexType, typename ALPHA, typename MA, typename MB>
void
traxpy(StorageOrder order, StorageUpLo upLo, Transpose trans, Diag diag,
       IndexType m, IndexType n, const ALPHA &alpha, const MA *A, IndexType ldA,
       MB *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("traxpy_generic");
    using std::min;


    if (order==RowMajor) {
        upLo  = (upLo==Upper) ? Lower : Upper;
        traxpy(ColMajor, upLo, trans, diag, n, m, alpha, A, ldA, B, ldB);
        return;
    }
    if (diag==NonUnit) {
        if (trans==NoTrans) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    axpy(min(j+1,m), alpha, A+j*ldA, IndexType(1),
                                     B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    axpy(m-j, alpha, A+j*(ldA+1), IndexType(1),
                              B+j*(ldB+1), IndexType(1));
                }
            }
        } else if (trans==Conj) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    acxpy(min(j+1,m), alpha, A+j*ldA, IndexType(1),
                                      B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    acxpy(m-j, alpha, A+j*(ldA+1), IndexType(1),
                               B+j*(ldB+1), IndexType(1));
                }
            }
        } else if (trans==Trans) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    axpy(min(j+1,m), alpha, A+j, ldA,
                                     B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    axpy(m-j, alpha, A+j*(ldA+1), ldA,
                              B+j*(ldB+1), IndexType(1));
                }
            }
        } else if (trans==ConjTrans) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    acxpy(min(j+1,m), alpha, A+j, ldA,
                                      B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    acxpy(m-j, alpha, A+j*(ldA+1), ldA,
                               B+j*(ldB+1), IndexType(1));
                }
            }
        } else {
            // unknown trans
            ASSERT(0);
        }
    } else { // diag==Unit

        // Not possible
        ASSERT(0);

    }
}

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_TRAXPY_TCC

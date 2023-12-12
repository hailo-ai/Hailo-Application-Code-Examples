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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_TRCOPY_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_TRCOPY_TCC 1

#include <algorithm>
#include <cassert>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

//
//  B = A  or B = A^T
//
template <typename IndexType, typename MA, typename MB>
void
trcopy(StorageOrder order, StorageUpLo upLo, Transpose trans, Diag diag,
       IndexType m, IndexType n, const MA *A, IndexType ldA,
       MB *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("trcopy_generic");
    using std::min;


    if (order==RowMajor) {
        upLo  = (upLo==Upper) ? Lower : Upper;
        trcopy(ColMajor, upLo, trans, diag, n, m, A, ldA, B, ldB);
        return;
    }
    if (diag==NonUnit) {
        if (trans==NoTrans) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    copy(min(j+1,m), A+j*ldA, IndexType(1),
                                     B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    copy(m-j, A+j*(ldA+1), IndexType(1),
                              B+j*(ldB+1), IndexType(1));
                }
            }
        } else if (trans==Conj) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    ccopy(min(j+1,m), A+j*ldA, IndexType(1),
                                      B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    ccopy(m-j, A+j*(ldA+1), IndexType(1),
                               B+j*(ldB+1), IndexType(1));
                }
            }
        } else if (trans==Trans) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    copy(min(j+1,m), A+j, ldA,
                                     B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    copy(m-j, A+j*(ldA+1), ldA,
                              B+j*(ldB+1), IndexType(1));
                }
            }
        } else if (trans==ConjTrans) {
            if (upLo==Upper) {
                for (IndexType j=0; j<n; ++j) {
                    ccopy(min(j+1,m), A+j, ldA,
                                      B+j*ldB, IndexType(1));
                }
            } else {
                for (IndexType j=0; j<min(m,n); ++j) {
                    ccopy(m-j, A+j*(ldA+1), ldA,
                               B+j*(ldB+1), IndexType(1));
                }
            }
        } else {
            // unknown trans
            ASSERT(0);
        }
    } else { // diag==Unit

        if (upLo==Upper) {
            if (trans==NoTrans || trans==Conj) {
                trcopy(order, upLo, trans, NonUnit,
                       m-1, n-1,
                       A+ldA, ldA, B+ldB, ldB);
            } else {
                trcopy(order, upLo, trans, NonUnit,
                       m-1, n-1,
                       A+1, ldA, B+ldB, ldB);
            }
        } else {
            if (trans==NoTrans || trans==Conj) {
                trcopy(order, upLo, trans, NonUnit,
                       m-1, n-1,
                       A+1, ldA, B+1, ldB);
            } else {
                trcopy(order, upLo, trans, NonUnit,
                       m-1, n-1,
                       A+ldA, ldA, B+1, ldB);
            }
        }
        return;

    }
}

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_TRCOPY_TCC

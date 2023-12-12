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

#ifndef CXXBLAS_TINYLEVEL1_GEAXPY_TCC
#define CXXBLAS_TINYLEVEL1_GEAXPY_TCC 1

#include "xflens/cxxblas/cxxblas.h"

#define HAVE_CXXBLAS_GEAXPY 1

namespace cxxblas {

template <int m, int n, typename ALPHA,
          typename MA, int ldA,
          typename MB, int ldB>
void
geaxpy(Transpose trans, const ALPHA &alpha, const MA *A, MB *B)
{

    CXXBLAS_DEBUG_OUT("geaxpy [tiny]");

    if (trans==NoTrans) {
        if ((ldA==n) && (ldB==n)) {
            axpy<m*n, ALPHA, MA, 1, MB, 1>(alpha, A, B);
            return;
        } else {
            for (int i=0, iA=0, iB=0; i<m; ++i, iA+=ldA, iB+=ldB) {
                axpy<n, ALPHA, MA, 1, MB, 1>(alpha, A+iA, B+iB);
            }
            return;
        }
    }
    if (trans==Conj) {
        if ((ldA==n) && (ldB==n)) {
            acxpy<m*n, ALPHA, MA, 1, MB, 1>(alpha, A, B);
            return;
        } else {
            for (int i=0, iA=0, iB=0; i<m; ++i, iA+=ldA, iB+=ldB) {
                acxpy<n, ALPHA, MA, 1, MB, 1>(alpha, A+iA, B+iB);
            }
            return;
        }
    }
     if (trans==Trans) {
        for (int i=0, iB=0; i<m; ++i, iB+=ldB) {
            axpy<n, ALPHA, MA, ldA, MB, 1>(alpha, A+i, B+iB);
        }
        return;
    }
    if (trans==ConjTrans) {
        for (int i=0, iB=0; i<m; ++i, iB+=ldB) {
            acxpy<n, ALPHA, MA, ldA, MB, 1>(alpha, A+i, B+iB);
        }
        return;
    }
 }

} // namespace cxxblas

#endif // CXXBLAS_TINYLEVEL1_GEAXPY_TCC

/*
 *   Copyright (c) 2012, Klaus Pototzky
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

#ifndef CXXBLAS_LEVEL3EXTENSION_GBMM_TCC
#define CXXBLAS_LEVEL3EXTENSION_GBMM_TCC 1

#include <complex>
#include "xflens/cxxblas/cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename VC>
void
gbmm(StorageOrder order, Side sideA,
      Transpose transA, Transpose transB,
      IndexType m, IndexType n,
      IndexType kl, IndexType ku,
      IndexType l,
      const ALPHA &alpha,
      const MA *A, IndexType ldA,
      const MB *B, IndexType ldB,
      const BETA &beta,
      VC *C, IndexType ldC)
{
    if (order==RowMajor) {
        if (sideA==Left) {
            if (transB==NoTrans) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, m, n, kl, ku,
                         alpha, A, ldA, B+i, ldB, beta, C+i, ldC);
                }
            } else if (transB==Conj) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, Conj, m, n, kl, ku,
                         alpha, A, ldA, B+i, ldB, beta, C+i, ldC);
                }
            } else if (transB==Trans) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, m, n, kl, ku,
                             alpha, A, ldA, B+i*ldB, IndexType(1),
                             beta, C+i, ldC);
                }
            } else if (transB==ConjTrans) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, Conj, m, n, kl, ku,
                         alpha, A, ldA, B+i*ldB, IndexType(1),
                         beta, C+i, ldC);
                }
            }
        } else if (sideA==Right) {
            transA = Transpose(transA^Trans);
            if (transB==NoTrans) {
                for (IndexType i=0; i<m; ++i) {
                    gbmv(RowMajor, transA, n, l, kl, ku,
                         alpha, A, ldA, B+i*ldB, IndexType(1),
                         beta, C+i*ldC, IndexType(1));
                }
            } else if (transB==Conj) {
                for (IndexType i=0; i<m; ++i) {
                    gbmv(RowMajor, transA, Conj, n, l, kl, ku,
                         alpha, A, ldA, B+i*ldB, IndexType(1),
                         beta, C+i*ldC, IndexType(1));
                }
            } else if (transB==Trans) {
                for (IndexType i=0; i<n; ++i) {
                    gbmv(RowMajor, transA, m, l, kl, ku,
                         alpha, A, ldA, B+i, ldB,
                         beta, C+i*ldC, IndexType(1));
                }
            } else if (transB==ConjTrans ) {
                for (IndexType i=0; i<n; ++i) {
                      gbmv(RowMajor, transA, Conj, m, l, kl, ku,
                           alpha, A, ldA, B+i, ldB,
                           beta, C+i*ldC, IndexType(1));
                }
            }
        }
    } else {
        if (sideA==Left) {
            transA = Transpose(transA^Trans);
            if (transB==NoTrans) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, n, m, ku, kl,
                         alpha, A, ldA, B+i*ldB, IndexType(1),
                         beta, C+i*ldC, IndexType(1));
                }
            } else if (transB==Conj) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, Conj, n, m, ku, kl,
                         alpha, A, ldA, B+i*ldB, IndexType(1),
                         beta, C+i*ldC, IndexType(1));
                }
            } else if (transB==Trans) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, n, m, ku, kl,
                         alpha, A, ldA, B+i, ldB,
                         beta, C+i*ldC, IndexType(1));
                }
            } else if (transB==ConjTrans) {
                for (IndexType i=0; i<l; ++i) {
                    gbmv(RowMajor, transA, Conj, n, m, ku, kl,
                         alpha, A, ldA, B+i, ldB,
                         beta, C+i*ldC, IndexType(1));
                }
            }
        } else if (sideA==Right) {
            if (transB==NoTrans) {
                for (IndexType i=0; i<m; ++i) {
                    gbmv(RowMajor, transA, l, n, ku, kl,
                         alpha, A, ldA, B+i, ldB, beta, C+i, ldC);
                }
            } else if (transB==Conj) {
                for (IndexType i=0; i<m; ++i) {
                    gbmv(RowMajor, transA, Conj, l, n, ku, kl,
                           alpha, A, ldA, B+i, ldB, beta, C+i, ldC);
                }
            } else if (transB==Trans) {
                for (IndexType i=0; i<n; ++i) {
                    gbmv(RowMajor, transA, l, m, ku, kl,
                           alpha, A, ldA, B+i*ldB, IndexType(1),
                           beta, C+i, ldC);
                }
            } else if (transB==ConjTrans ) {
                for (IndexType i=0; i<n; ++i) {
                    gbmv(RowMajor, transA, Conj, l, m, ku, kl,
                         alpha, A, ldA, B+i*ldB, IndexType(1), beta, C+i,ldC);
                }
            }
        }
    }
    return;
}


} // namespace cxxblas

#endif // CXXBLAS_LEVEL3EXTENSION_GBMM_TCC

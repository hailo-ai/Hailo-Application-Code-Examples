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

#ifndef CXXBLAS_DRIVERS_DRIVERS_TCC
#define CXXBLAS_DRIVERS_DRIVERS_TCC 1

#include "xflens/cxxblas/auxiliary/auxiliary.h"
#include "xflens/cxxblas/drivers/drivers.h"

#ifndef BLAS_IMPL
#   define BLAS_IMPL    "CXXBLAS (generic)"
#endif

namespace cxxblas {

template <typename CHAR>
const CHAR *
blasImpl()
{
    return BLAS_IMPL;
}

//------------------------------------------------------------------------------
template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Transpose>::value, char>::Type
getF77BlasChar(ENUM trans)
{
    if (trans==NoTrans) {
        return 'N';
    } else if (trans==Trans) {
        return 'T';
    } else if (trans==Conj) {
        return 'R';
    } else if (trans==ConjTrans) {
        return 'C';
    } else {
        ASSERT(0);
        return '?';
    }
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Diag>::value, char>::Type
getF77BlasChar(ENUM diag)
{
    if (diag==Unit) {
        return 'U';
    }
    return 'N';
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, char>::Type
getF77BlasChar(ENUM upLo)
{
    if (upLo==Upper) {
        return 'U';
    }
    return 'L';
}

//------------------------------------------------------------------------------
template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Transpose>::value, Transpose>::Type
getCxxBlasEnum(char trans)
{
    if ((trans=='N') || (trans=='n')) {
        return NoTrans;
    } else if ((trans=='T') || (trans=='t')) {
        return Trans;
    } else if ((trans=='C') || (trans=='c')) {
        return ConjTrans;
    } else if ((trans=='R') || (trans=='r')) {
        return Conj;
    }
    ASSERT(0);
    return NoTrans;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Diag>::value, Diag>::Type
getCxxBlasEnum(char diag)
{
    if (diag=='U') {
        return Unit;
    }
    ASSERT(diag=='N');
    return NonUnit;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, StorageUpLo>::Type
getCxxBlasEnum(char upLo)
{
    if (upLo=='U') {
        return Upper;
    }
    ASSERT(upLo=='L');
    return Lower;
}

//------------------------------------------------------------------------------
#ifdef HAVE_CBLAS

namespace CBLAS {

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageOrder>::value, CBLAS_ORDER>::Type
getCblasType(ENUM order)
{
    if (order==RowMajor) {
        return CblasRowMajor;
    }
    return CblasColMajor;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Transpose>::value, CBLAS_TRANSPOSE>::Type
getCblasType(ENUM trans)
{
    if (trans==NoTrans) {
        return CblasNoTrans;
    }
    if (trans==Conj) {
        return CblasConjNoTrans;
    }
    if (trans==Trans) {
        return CblasTrans;
    }
    return CblasConjTrans;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, CBLAS_UPLO>::Type
getCblasType(ENUM upLo)
{
    if (upLo==Upper) {
        return CblasUpper;
    }
    return CblasLower;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Side>::value, CBLAS_SIDE>::Type
getCblasType(ENUM side)
{
    if (side==Left) {
        return CblasLeft;
    }
    return CblasRight;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Diag>::value, CBLAS_DIAG>::Type
getCblasType(ENUM diag)
{
    if (diag==Unit) {
        return CblasUnit;
    }
    return CblasNonUnit;
}

} // namespace CBLAS

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_DRIVERS_DRIVERS_TCC

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

#ifndef CXXBLAS_DRIVERS_DRIVERS_H
#define CXXBLAS_DRIVERS_DRIVERS_H 1

#include "xflens/cxxblas/auxiliary/issame.h"
#include "xflens/cxxblas/auxiliary/restrictto.h"

// define implementation specific constants, macros, etc.
#if defined (WITH_ATLAS)
#   include "xflens/cxxblas/drivers/atlas.h"
#elif defined (WITH_GOTOBLAS)
#   include "xflens/cxxblas/drivers/gotoblas.h"
#elif defined (WITH_OPENBLAS)
#   include "xflens/cxxblas/drivers/openblas.h"
#elif defined (WITH_VECLIB)
#   include "xflens/cxxblas/drivers/veclib.h"
#elif defined (WITH_MKLBLAS)
#   include "xflens/cxxblas/drivers/mklblas.h"
#elif defined (WITH_REFBLAS)
#   include "xflens/cxxblas/drivers/refblas.h"
#endif


#ifdef HAVE_CBLAS
#include "xflens/cxxblas/drivers/cblas.h"
#endif

#ifdef HAVE_SPARSEBLAS
#include "xflens/cxxblas/drivers/sparseblas.h"
#endif

#include "xflens/cxxblas/typedefs.h"

namespace cxxblas {

template <typename CHAR>
const CHAR *
blasImpl();

//------------------------------------------------------------------------------

template <typename Any>
struct If
{
};

template <>
struct If<int>
{
    typedef void isBlasCompatibleInteger;
};

template <>
struct If<long>
{
    typedef void isBlasCompatibleInteger;
};

//------------------------------------------------------------------------------
template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Transpose>::value, char>::Type
    getF77BlasChar(ENUM trans);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Diag>::value, char>::Type
    getF77BlasChar(ENUM diag);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, char>::Type
    getF77BlasChar(ENUM upLo);

//------------------------------------------------------------------------------
template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Transpose>::value, Transpose>::Type
    getCxxBlasEnum(char trans);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Diag>::value, Diag>::Type
    getCxxBlasEnum(char diag);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, StorageUpLo>::Type
    getCxxBlasEnum(char upLo);

//------------------------------------------------------------------------------
#ifdef HAVE_CBLAS

namespace CBLAS {

//TODO: rename these to getCblasEnum

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageOrder>::value, CBLAS_ORDER>::Type
    getCblasType(ENUM order);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Transpose>::value, CBLAS_TRANSPOSE>::Type
    getCblasType(ENUM trans);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, CBLAS_UPLO>::Type
    getCblasType(ENUM upLo);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Side>::value, CBLAS_SIDE>::Type
    getCblasType(ENUM side);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Diag>::value, CBLAS_DIAG>::Type
    getCblasType(ENUM diag);

} // namespace CBLAS

#endif // HAVE_CBLAS

} // namespace cxxblas


#endif // CXXBLAS_DRIVERS_DRIVERS_H

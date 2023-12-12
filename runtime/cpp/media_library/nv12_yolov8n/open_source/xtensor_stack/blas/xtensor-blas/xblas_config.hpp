/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_CONFIG_HPP
#define XBLAS_CONFIG_HPP

#define XTENSOR_BLAS_VERSION_MAJOR 0
#define XTENSOR_BLAS_VERSION_MINOR 20
#define XTENSOR_BLAS_VERSION_PATCH 0

#ifndef XTENSOR_USE_FLENS_BLAS
#define HAVE_CBLAS 1
#endif

#ifndef USE_CXXLAPACK
#define USE_CXXLAPACK
#endif

#ifndef BLAS_IDX
#define BLAS_IDX int
#endif

#ifdef __CLING__
#include "xtensor-blas/xblas_config_cling.hpp"
#endif

namespace xt
{
    using blas_index_t = BLAS_IDX;
}

#endif

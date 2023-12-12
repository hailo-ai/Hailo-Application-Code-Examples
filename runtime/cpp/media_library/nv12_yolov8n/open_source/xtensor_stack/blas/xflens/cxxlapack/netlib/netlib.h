#ifndef CXXLAPACK_NETLIB_NETLIB_H
#define CXXLAPACK_NETLIB_NETLIB_H 1

#ifdef LAPACK_IMPL
#   undef   LAPACK_IMPL
#endif

#ifndef CXXLAPACK_NO_UNDERSCORE
#   define     LAPACK_IMPL(x)           x##_
#else
#   define     LAPACK_IMPL(x)           x
#endif

#ifdef  DEBUG_CXXLAPACK
#define CXXLAPACK_DEBUG_OUT(msg) std::cerr << "CXXLAPACK: " << msg << std::endl;
#endif

#ifndef CXXLAPACK_DEBUG_OUT
#   define  CXXLAPACK_DEBUG_OUT(msg)
#endif

namespace cxxlapack {

extern "C" {
#   include "xflens/cxxlapack/netlib/interface/lapack.in.h"
} // extern "C"

}

#endif //  CXXLAPACK_NETLIB_NETLIB_H

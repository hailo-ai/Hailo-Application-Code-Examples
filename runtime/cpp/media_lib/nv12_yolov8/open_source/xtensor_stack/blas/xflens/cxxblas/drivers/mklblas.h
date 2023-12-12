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

#ifndef CXXBLAS_DRIVERS_MKLBLAS_H
#define CXXBLAS_DRIVERS_MKLBLAS_H 1


#   define HAVE_CBLAS           1
#   define HAVE_SPARSEBLAS      1
#   define WITH_MKLDSS          1
#   ifdef MKL_ILP64
#      define CBLAS_INT         long
#      define CBLAS_INDEX       long
#   else
#      define CBLAS_INT         int
#      define CBLAS_INDEX       int
#   endif
#   define BLAS_IMPL            "MKLBLAS"

// BLAS extensions
#ifndef HAVE_CBLAS_AXPBY
#    define HAVE_CBLAS_AXPBY
#    define BLAS_EXT(x)         cblas_##x
#endif

// MKL includes LAPACK
#ifndef USE_CXXLAPACK
#    define USE_CXXLAPACK       1
#endif
// MKL includes FFTW interface (float, double)
#ifndef HAVE_FFTW
#    define HAVE_FFTW           1
#endif
#ifndef HAVE_FFTW_FLOAT
#    define HAVE_FFTW_FLOAT     1
#endif
#ifndef HAVE_FFTW_DOUBLE
#    define HAVE_FFTW_DOUBLE    1
#endif

#endif // CXXBLAS_DRIVERS_MKLBLAS_H

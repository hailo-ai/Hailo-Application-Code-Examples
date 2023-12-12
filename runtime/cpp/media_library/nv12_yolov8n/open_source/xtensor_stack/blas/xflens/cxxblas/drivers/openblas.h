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

#ifndef CXXBLAS_DRIVERS_OPENBLAS_H
#define CXXBLAS_DRIVERS_OPENBLAS_H 1

#   define HAVE_CBLAS           1
#   ifdef BLASINT
#      define CBLAS_INT         BLASINT
#   else
#      define CBLAS_INT         int
#   endif
#   define BLAS_IMPL            "OpenBLAS"
#   ifndef CBLAS_INDEX
#       define CBLAS_INDEX      size_t
#   endif // CBLAS_INDEX

// BLAS extensions
#ifndef HAVE_CBLAS_AXPBY
#    define HAVE_CBLAS_AXPBY
#    define BLAS_EXT(x)         cblas_##x
#endif

extern "C" {
	/* Assume C declarations for C++ */

	/*Set the number of threads on runtime.*/
	void openblas_set_num_threads(int num_threads);
	void goto_set_num_threads(int num_threads);

	/*Get the number of threads on runtime.*/
	int openblas_get_num_threads(void);

	/*Get the number of physical processors (cores).*/
	int openblas_get_num_procs(void);

	/*Get the build configure on runtime.*/
	char* openblas_get_config(void);

	/*Get the CPU corename on runtime.*/
	char* openblas_get_corename(void);

	/* Get the parallelization type which is used by OpenBLAS */
	int openblas_get_parallel(void);
}

/* OpenBLAS is compiled for sequential use  */
#define OPENBLAS_SEQUENTIAL  0
/* OpenBLAS is compiled using normal threading model */
#define OPENBLAS_THREAD  1
/* OpenBLAS is compiled using OpenMP threading model */
#define OPENBLAS_OPENMP 2

#endif // CXXBLAS_DRIVERS_OPENBLAS_H

/*
 *   Copyright (c) 2009, Michael Lehn
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

#ifndef CXXBLAS_AUXILIARY_DEBUGMACRO_H
#define CXXBLAS_AUXILIARY_DEBUGMACRO_H 1

#include <iostream>


//-- CXXBLAS_DEBUG_OUT ---------------------------------------------------------
#ifdef CXXBLAS_DEBUG
#   ifndef CXXBLAS_DEBUG_OUT
#   define CXXBLAS_DEBUG_OUT(msg)  std::cerr << "CXXBLAS: " << msg << std::endl
#   endif // CXXBLAS_DEBUG_OUT
#else
#   ifndef CXXBLAS_DEBUG_OUT
#   define CXXBLAS_DEBUG_OUT(msg)
#   endif // CXXBLAS_DEBUG_OUT
#endif // CXXBLAS_DEBUG

#include <cassert>

//-- ASSERT -------------------------------------------------------------------

// ASSERT which prints out a trace back of the call
#if defined(TRACEBACK_ASSERT) && !defined(NDEBUG)
#   include <execinfo.h>
#   ifndef ASSERT
#       define ASSERT(x) if(!(x)) {                                            \
                             void* callstack[128];                             \
                             int frames = backtrace(callstack, 128);           \
                             char** strs = backtrace_symbols(callstack,        \
                                                             frames);          \
                             for (int i=0; i<frames; ++i) {                    \
                                 std::cerr << strs[i] << std::endl;            \
                             }                                                 \
                             free(strs);                                       \
                         }                                                     \
                         assert(x);
#   endif
#endif

// Default ASSERT Macro
#ifndef ASSERT
#   define ASSERT(x) assert(x)
#endif

// Prevent warings because some function parameters are only used in debug
// mode within assertions.  In non-debug mode this causes warnings because
// of unused variables.

#ifndef NDEBUG

#   ifndef DEBUG_VAR
#   define DEBUG_VAR(x)      x
#   endif

#   ifndef FAKE_USE_NDEBUG
#   define FAKE_USE_NDEBUG(x)
#   endif

#else

#   ifndef FAKE_USE_NDEBUG
#   define FAKE_USE_NDEBUG(x) (void)x
#   endif

#endif


#endif // CXXBLAS_AUXILIARY_DEBUGMACRO_H

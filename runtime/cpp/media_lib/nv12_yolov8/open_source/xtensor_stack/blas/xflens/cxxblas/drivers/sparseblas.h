/*
 *   Modification of mkl_spblas.h
 */

#ifndef CXXBLAS_DRIVERS_SPARSEBLAS_H
#define CXXBLAS_DRIVERS_SPARSEBLAS_H 1

#include "xflens/cxxblas/drivers/mklblas.h"

#ifndef SPARSEBLAS_INT
#   define SPARSEBLAS_INT int
#endif // CBLAS_INT

#ifndef SPARSEBLAS_INDEX
#   define SPARSEBLAS_INDEX int
#endif // CBLAS_INT


#ifdef __cplusplus
extern "C" {
#endif

//-- LEVEL 2 -------------------------------------------------------------------


// mv
void mkl_scscmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                const float *beta,
                float *y);

void mkl_scsrmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                const float *beta,
                float *y);

void mkl_dcscmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                const double *beta,
                double *y);

void mkl_dcsrmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                const double *beta,
                double *y);

void mkl_ccscmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                const float *beta,
                float *y);

void mkl_ccsrmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                const float *beta,
                float *y);

void mkl_zcscmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                const double *beta,
                double *y);

void mkl_zcsrmv(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                const double *beta,
                double *y);

// sv
void mkl_scsrsv(const char *transa,
                const CBLAS_INT *m,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                float *y);

void mkl_scscsv(const char *transa,
                const CBLAS_INT *m,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                float *y);

void mkl_dcsrsv(const char *transa,
                const CBLAS_INT *m,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                double *y);

void mkl_dcscsv(const char *transa,
                const CBLAS_INT *m,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                double *y);

void mkl_ccsrsv(const char *transa,
                const CBLAS_INT *m,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                float *y);

void mkl_ccscsv(const char *transa,
                const CBLAS_INT *m,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *x,
                float *y);

void mkl_zcsrsv(const char *transa,
                const CBLAS_INT *m,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                double *y);

void mkl_zcscsv(const char *transa,
                const CBLAS_INT *m,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *x,
                double *y);

// Level 3

// mm
void mkl_scscmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const float *alpha,
                char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                CBLAS_INT *ldb,
                const float *beta,
                const float *c,
                CBLAS_INT *ldc);

void mkl_scsrmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                const float *beta,
                float *c,
                const CBLAS_INT *ldc);

void mkl_dcscmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                const double *beta,
                double *c,
                const CBLAS_INT *ldc);

void mkl_dcsrmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                const double *beta,
                double *c,
                const CBLAS_INT *ldc);

void mkl_ccscmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                const float *beta,
                float *c,
                const CBLAS_INT *ldc);

void mkl_ccsrmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                const float *beta,
                float *c,
                const CBLAS_INT *ldc);

void mkl_zcscmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                const double *beta,
                double *c,
                const CBLAS_INT *ldc);

void mkl_zcsrmm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const CBLAS_INT *k,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                const double *beta,
                double *c,
                const CBLAS_INT *ldc);

// sm
void mkl_scsrsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                float *c,
                const CBLAS_INT *ldc);

void mkl_scscsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                float *c,
                const CBLAS_INT *ldc);

void mkl_dcsrsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                double *c,
                const CBLAS_INT *ldc);

void mkl_dcscsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                double *c,
                const CBLAS_INT *ldc);

void mkl_ccsrsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                float *c,
                const CBLAS_INT *ldc);

void mkl_ccscsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const float *alpha,
                const char *matdescra,
                const float  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const float *b,
                const CBLAS_INT *ldb,
                float *c,
                const CBLAS_INT *ldc);

void mkl_zcsrsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                double *c,
                const CBLAS_INT *ldc);

void mkl_zcscsm(const char *transa,
                const CBLAS_INT *m,
                const CBLAS_INT *n,
                const double *alpha,
                const char *matdescra,
                const double  *val,
                const CBLAS_INT *indx,
                const CBLAS_INT *pntrb,
                const CBLAS_INT *pntre,
                const double *b,
                const CBLAS_INT *ldb,
                double *c,
                const CBLAS_INT *ldc);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CXXBLAS_DRIVERS_SPARSEBLAS_H

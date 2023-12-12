/*
 *   File taken (with minor modifications) from cblas:
 *     http://www.netlib.org/blas/
 */

#ifndef CXXBLAS_DRIVERS_CBLAS_H
#define CXXBLAS_DRIVERS_CBLAS_H 1

#ifndef CBLAS_INT
#   define CBLAS_INT int
#endif // CBLAS_INT

#ifndef CBLAS_INDEX
#   define CBLAS_INDEX int
#endif // CBLAS_INT


#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER        {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE    {CblasNoTrans=111, CblasTrans=112,
                         CblasConjTrans=113, CblasConjNoTrans=114};
enum CBLAS_UPLO         {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG         {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE         {CblasLeft=141, CblasRight=142};

//-- LEVEL 1 -------------------------------------------------------------------

// asum
float
cblas_sasum(CBLAS_INT n, const float *x, CBLAS_INT incX);

double
cblas_dasum (CBLAS_INT n, const double *x, CBLAS_INT incX);

float
cblas_scasum(CBLAS_INT n, const float *x, CBLAS_INT incX);

double
cblas_dzasum(CBLAS_INT n, const double *x, CBLAS_INT incX);

// axpy
void
cblas_saxpy(CBLAS_INT n, float alpha,
            const float *x, CBLAS_INT incX,
            float *y, CBLAS_INT incY);

void
cblas_daxpy(CBLAS_INT n, double alpha,
            const double *x, CBLAS_INT incX,
            double *y, CBLAS_INT incY);

void
cblas_caxpy(CBLAS_INT n, const float *alpha,
            const float *x, CBLAS_INT incX,
            float *y, CBLAS_INT incY);

void
cblas_zaxpy(CBLAS_INT n, const double *alpha,
            const double *x, CBLAS_INT incX,
            double *y, CBLAS_INT incY);

// axpby
#ifdef HAVE_CBLAS_AXPBY

void
BLAS_EXT(saxpby) (CBLAS_INT n, float alpha,
                 const float *x, CBLAS_INT incX,
                 float beta, float *y, CBLAS_INT incY);

void
BLAS_EXT(daxpby) (CBLAS_INT n, double alpha,
                const double *x, CBLAS_INT incX,
                double beta, double *y, CBLAS_INT incY);

void
BLAS_EXT(caxpby) (CBLAS_INT n, const float *alpha,
                 const float *x, CBLAS_INT incX,
                 const float *beta, float *y, CBLAS_INT incY);

void
BLAS_EXT(zaxpby) (CBLAS_INT n, const double *alpha,
                 const double *x, CBLAS_INT incX,
                 const double *beta, double *y, CBLAS_INT incY);

#endif //HAVE_CBLAS_AXPBY

// copy
void
cblas_scopy(CBLAS_INT n,
            const float *x, CBLAS_INT incX,
            float *y, CBLAS_INT incY);

void
cblas_dcopy(CBLAS_INT n,
            const double *x, CBLAS_INT incX,
            double *y, CBLAS_INT incY);

void
cblas_ccopy(CBLAS_INT n,
            const float *x, CBLAS_INT incX,
            float *y, CBLAS_INT incY);

void
cblas_zcopy(CBLAS_INT n,
            const double *x, CBLAS_INT incX,
            double *y, CBLAS_INT incY);

// dot
float
cblas_sdsdot(CBLAS_INT n, float alpha,
             const float *x, CBLAS_INT incX,
             const float *y, CBLAS_INT incY);

double
cblas_dsdot(CBLAS_INT n,
            const float *x, CBLAS_INT incX,
            const float *y, CBLAS_INT incY);

float
cblas_sdot(CBLAS_INT n,
           const float *x, CBLAS_INT incX,
           const float  *y, CBLAS_INT incY);

double
cblas_ddot(CBLAS_INT n,
           const double *x, CBLAS_INT incX,
           const double *y, CBLAS_INT incY);

void
cblas_cdotu_sub(CBLAS_INT n,
                const float  *x, CBLAS_INT incX,
                const float  *y, CBLAS_INT incY,
                float *result);

void
cblas_cdotc_sub(CBLAS_INT n,
                const float  *x, CBLAS_INT incX,
                const float  *y, CBLAS_INT incY,
                float *result);

void
cblas_zdotu_sub(CBLAS_INT n,
                const double *x, CBLAS_INT incX,
                const double *y, CBLAS_INT incY,
                double *result);

void
cblas_zdotc_sub(CBLAS_INT n,
                const double *x, CBLAS_INT incX,
                const double *y, CBLAS_INT incY,
                double *result);

// iamax
CBLAS_INDEX
cblas_isamax(CBLAS_INT n, const float *x, CBLAS_INT incX);

CBLAS_INDEX
cblas_idamax(CBLAS_INT n, const double *x, CBLAS_INT incX);

CBLAS_INDEX
cblas_icamax(CBLAS_INT n, const float *x, CBLAS_INT incX);

CBLAS_INDEX
cblas_izamax(CBLAS_INT n, const double *x, CBLAS_INT incX);

// nrm2
float
cblas_snrm2(CBLAS_INT n, const float *X, CBLAS_INT incX);

double
cblas_dnrm2(CBLAS_INT n, const double *X, CBLAS_INT incX);

float
cblas_scnrm2(CBLAS_INT n, const float *X, CBLAS_INT incX);

double
cblas_dznrm2(CBLAS_INT n, const double *X, CBLAS_INT incX);

// rot
void
cblas_srot(CBLAS_INT n,
           float *X, CBLAS_INT incX,
           float *Y, CBLAS_INT incY,
           float c, float s);

void
cblas_drot(CBLAS_INT n,
           double *X, CBLAS_INT incX,
           double *Y, CBLAS_INT incY,
           double c, double  s);

void
cblas_srotg(float *a, float *b, float *c, float *s);

void
cblas_drotg(double *a, double *b, double *c, double *s);

// rotm
void
cblas_srotm(CBLAS_INT n,
            float *X, CBLAS_INT incX,
            float *Y, CBLAS_INT incY,
            const float *P);

void
cblas_drotm(CBLAS_INT n,
            double *X, CBLAS_INT incX,
            double *Y, CBLAS_INT incY,
            const double *P);

void
cblas_srotmg(float *d1, float *d2, float *b1, float *b2, float *P);

void
cblas_drotmg(double *d1, double *d2, double *b1, double *b2, double *P);

// scal
void
cblas_sscal(CBLAS_INT n, float alpha, float *x, CBLAS_INT incX);

void
cblas_dscal(CBLAS_INT n, double alpha, double *x, CBLAS_INT incX);

void
cblas_cscal(CBLAS_INT n, const float *alpha, float *x, CBLAS_INT incX);

void
cblas_zscal(CBLAS_INT n, const double *alpha, double *x, CBLAS_INT incX);

void
cblas_csscal(CBLAS_INT n, float alpha, float *x, CBLAS_INT incX);

void
cblas_zdscal(CBLAS_INT n, double alpha, double *x, CBLAS_INT incX);

// swap
void
cblas_sswap(CBLAS_INT n, float *x, CBLAS_INT incX, float *y, CBLAS_INT incY);

void
cblas_dswap(CBLAS_INT n, double *x, CBLAS_INT incX, double *y, CBLAS_INT incY);

void
cblas_cswap(CBLAS_INT n, float *x, CBLAS_INT incX, float *y, CBLAS_INT incY);

void
cblas_zswap(CBLAS_INT n, double *x, CBLAS_INT incX, double *y, CBLAS_INT incY);

//-- LEVEL 2 -------------------------------------------------------------------

// gbmv
void
cblas_sgbmv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            CBLAS_INT kl, CBLAS_INT ku,
            float alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            float beta,
            float *y, CBLAS_INT incY);

void
cblas_dgbmv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            CBLAS_INT kl, CBLAS_INT ku,
            double alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            double beta,
            double *y, CBLAS_INT incY);

void
cblas_cgbmv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            CBLAS_INT kl, CBLAS_INT ku,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            const float *beta,
            float *y, CBLAS_INT incY);

void
cblas_zgbmv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            CBLAS_INT kl, CBLAS_INT ku,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            const double *beta,
            double *y, CBLAS_INT incY);

// gemv
void
cblas_sgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            float alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            float beta,
            float *y, CBLAS_INT incY);

void
cblas_dgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            double alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            double beta,
            double *y, CBLAS_INT incY);

void
cblas_cgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            const float *beta,
            float *y, CBLAS_INT incY);

void
cblas_zgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            const double *beta,
            double *y, CBLAS_INT incY);

// sbmv
void
cblas_ssbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, CBLAS_INT k, float alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            float beta,
            float *y, CBLAS_INT incY);

void
cblas_dsbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,CBLAS_INT k, double alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            double beta,
            double *y, CBLAS_INT incY);

// symv
void
cblas_ssymv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, float alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            float beta,
            float *y, CBLAS_INT incY);

void
cblas_dsymv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, double alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            double beta,
            double *y, CBLAS_INT incY);

// spmv
void
cblas_sspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, float alpha,
            const float *Ap,
            const float *x, CBLAS_INT incX,
            float beta,
            float *y, CBLAS_INT incY);

void
cblas_dspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, double alpha,
            const double *Ap,
            const double *x, CBLAS_INT incX,
            double beta,
            double *y, CBLAS_INT incY);

// hbmv
void
cblas_chbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, CBLAS_INDEX k, const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            const float *beta,
            float *y, CBLAS_INT incY);

void
cblas_zhbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, CBLAS_INDEX k, const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            const double *beta,
            double *y, CBLAS_INT incY);


// hemv
void
cblas_chemv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *x, CBLAS_INT incX,
            const float *beta,
            float *y, CBLAS_INT incY);

void
cblas_zhemv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *x, CBLAS_INT incX,
            const double *beta,
            double *y, CBLAS_INT incY);

// hpmv
void
cblas_chpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, const float *alpha,
            const float *Ap,
            const float *x, CBLAS_INT incX,
            const float *beta,
            float *y, CBLAS_INT incY);

void
cblas_zhpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n, const double *alpha,
            const double *Ap,
            const double *x, CBLAS_INT incX,
            const double *beta,
            double *y, CBLAS_INT incY);

// tbsv
void
cblas_stbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INT k,
            const float *A, CBLAS_INT lda,
            float *X, CBLAS_INT incX);

void
cblas_dtbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INT k,
            const double *A, CBLAS_INT lda,
            double *X, CBLAS_INT incX);

void
cblas_ctbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INT k,
            const float *A, CBLAS_INT lda,
            float *X, CBLAS_INT incX);

void
cblas_ztbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INT k,
            const double *A, CBLAS_INT lda,
            double *X, CBLAS_INT incX);

// trsv
void
cblas_strsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *A, CBLAS_INT lda,
            float *X, CBLAS_INT incX);

void
cblas_dtrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const double *A, CBLAS_INT lda,
            double *X, CBLAS_INT incX);

void
cblas_ctrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *A, CBLAS_INT lda,
            float *X, CBLAS_INT incX);

void
cblas_ztrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const double *A, CBLAS_INT lda,
            double *X, CBLAS_INT incX);

// tpsv
void
cblas_stpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *A,
            float *X, CBLAS_INT incX);

void
cblas_dtpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const double *A,
            double *X, CBLAS_INT incX);

void
cblas_ctpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *A,
            float *X, CBLAS_INT incX);

void
cblas_ztpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const double *A,
            double *X, CBLAS_INT incX);

// tbmv
void
cblas_stbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INDEX k,
            const float *A, CBLAS_INT lda,
            float *x, CBLAS_INT incX);

void
cblas_dtbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INDEX k,
            const double *A, CBLAS_INT lda,
            double *x, CBLAS_INT incX);

void
cblas_ctbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n, CBLAS_INDEX k,
            const float *A, CBLAS_INT lda,
            float *x, CBLAS_INT incX);

void
cblas_ztbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT N, CBLAS_INDEX k,
            const double *A, CBLAS_INT lda,
            double *x, CBLAS_INT incX);

// trmv
void
cblas_strmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *A, CBLAS_INT lda,
            float *x, CBLAS_INT incX);

void
cblas_dtrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const double *A, CBLAS_INT lda,
            double *x, CBLAS_INT incX);

void
cblas_ctrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *A, CBLAS_INT lda,
            float *x, CBLAS_INT incX);

void
cblas_ztrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT N,
            const double *A, CBLAS_INT lda,
            double *x, CBLAS_INT incX);

// tpmv
void
cblas_stpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *Ap,
            float *x, CBLAS_INT incX);

void
cblas_dtpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const double *Ap,
            double *x, CBLAS_INT incX);

void
cblas_ctpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT n,
            const float *Ap,
            float *x, CBLAS_INT incX);

void
cblas_ztpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT N,
            const double *Ap,
            double *x, CBLAS_INT incX);

// ger
void
cblas_sger(enum CBLAS_ORDER order,
           CBLAS_INT m, CBLAS_INT n,
           float alpha,
           const float *X, CBLAS_INT incX,
           const float *Y, CBLAS_INT incY,
           float *A, CBLAS_INT lda);

void
cblas_dger(enum CBLAS_ORDER order,
           CBLAS_INT m, CBLAS_INT n,
           double alpha,
           const double *X, CBLAS_INT incX,
           const double *Y, CBLAS_INT incY,
           double *A, CBLAS_INT lda);

void
cblas_cgeru(enum CBLAS_ORDER order,
            CBLAS_INT m, CBLAS_INT n,
            const float *alpha,
            const float *X, CBLAS_INT incX,
            const float *Y, CBLAS_INT incY,
            float  *A, CBLAS_INT lda);

void
cblas_cgerc(enum CBLAS_ORDER order,
            CBLAS_INT m, CBLAS_INT n,
            const float  *alpha,
            const float  *X, CBLAS_INT incX,
            const float  *Y, CBLAS_INT incY,
            float  *A, CBLAS_INT lda);

void
cblas_zgeru(enum CBLAS_ORDER order,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *X, CBLAS_INT incX,
            const double *Y, CBLAS_INT incY,
            double *A, CBLAS_INT lda);

void
cblas_zgerc(enum CBLAS_ORDER order,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *X, CBLAS_INT incX,
            const double *Y, CBLAS_INT incY,
            double *A, CBLAS_INT lda);

// syr
void
cblas_ssyr(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           float alpha,
           const float *X, CBLAS_INT incX,
           float *A, CBLAS_INT lda);

void
cblas_dsyr(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           double alpha,
           const double *X, CBLAS_INT incX,
           double *A, CBLAS_INT lda);

// spr
void
cblas_sspr(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           float alpha,
           const float *X, CBLAS_INT incX,
           float *A);

void
cblas_dspr(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           double alpha,
           const double *X, CBLAS_INT incX,
           double *A);

// her
void
cblas_cher(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           float alpha,
           const float *X, CBLAS_INT incX,
           float *A, CBLAS_INT lda);

void
cblas_zher(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           double alpha,
           const double *X, CBLAS_INT incX,
           double *A, CBLAS_INT lda);

// hpr
void
cblas_chpr(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           float alpha,
           const float *X, CBLAS_INT incX,
           float *A);

void
cblas_zhpr(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
           CBLAS_INT n,
           double alpha,
           const double *X, CBLAS_INT incX,
           double *A);

// spr2
void
cblas_sspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            float alpha,
            const float *X, CBLAS_INT incX,
            const float *Y, CBLAS_INT incY,
            float *A);

void
cblas_dspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            double alpha,
            const double *X, CBLAS_INT incX,
            const double *Y, CBLAS_INT incY,
            double *A);

// syr2
void
cblas_ssyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            float alpha,
            const float *X, CBLAS_INT incX,
            const float *Y, CBLAS_INT incY,
            float *A, CBLAS_INT lda);

void
cblas_dsyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            double alpha,
            const double *X, CBLAS_INT incX,
            const double *Y, CBLAS_INT incY,
            double *A, CBLAS_INT lda);
// her2
void
cblas_cher2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            const float *alpha,
            const float *X, CBLAS_INT incX,
            const float *Y, CBLAS_INT incY,
            float *A, CBLAS_INT lda);

void
cblas_zher2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            const double *alpha,
            const double *X, CBLAS_INT incX,
            const double *Y, CBLAS_INT incY,
            double *A, CBLAS_INT lda);
// hpr2
void
cblas_chpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            const float *alpha,
            const float *X, CBLAS_INT incX,
            const float *Y, CBLAS_INT incY,
            float *A);

void
cblas_zhpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            CBLAS_INT n,
            const double *alpha,
            const double *X, CBLAS_INT incX,
            const double *Y, CBLAS_INT incY,
            double *A);

//-- LEVEL 3 -------------------------------------------------------------------

// gemm
void
cblas_sgemm(enum CBLAS_ORDER order,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
            CBLAS_INT m, CBLAS_INT n, CBLAS_INT k,
            float alpha,
            const float *A, CBLAS_INT ldA,
            const float *B, CBLAS_INT ldB,
            float beta,
            float *C, CBLAS_INT ldC);

void
cblas_dgemm(enum CBLAS_ORDER order,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
            CBLAS_INT m, CBLAS_INT n, CBLAS_INT k,
            double alpha,
            const double *A, CBLAS_INT ldA,
            const double *B, CBLAS_INT ldB,
            double beta,
            double *C, CBLAS_INT ldC);

void
cblas_cgemm(enum CBLAS_ORDER order,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
            CBLAS_INT m, CBLAS_INT n, CBLAS_INT k,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *B, CBLAS_INT ldB,
            const float *beta,
            float *C, CBLAS_INT ldC);

void
cblas_zgemm(enum CBLAS_ORDER order,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
            CBLAS_INT m, CBLAS_INT n, CBLAS_INT k,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *B, CBLAS_INT ldB,
            const double *beta,
            double *C, CBLAS_INT ldC);

// hemm
void
cblas_chemm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            CBLAS_INT m, CBLAS_INT n,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *B, CBLAS_INT ldB,
            const float *beta,
            float *C, CBLAS_INT ldC);

void
cblas_zhemm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *B, CBLAS_INT ldB,
            const double *beta,
            double *C, CBLAS_INT ldC);

// herk
void
cblas_cherk(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE trans, CBLAS_INT n, CBLAS_INT k,
            float alpha,
            const float *A, CBLAS_INT ldA,
            float beta,
            float *C, CBLAS_INT ldC);

void
cblas_zherk(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE trans,
            CBLAS_INT n, CBLAS_INT k,
            double alpha,
            const double *A, CBLAS_INT ldA,
            double beta,
            double *C, CBLAS_INT ldC);

// her2k
void
cblas_cher2k(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
             enum CBLAS_TRANSPOSE trans,
             CBLAS_INT n, CBLAS_INT k,
             const float *alpha,
             const float *A, CBLAS_INT ldA,
             const float *B, CBLAS_INT ldB,
             float beta,
             float *C, CBLAS_INT ldC);

void
cblas_zher2k(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
             enum CBLAS_TRANSPOSE trans,
             CBLAS_INT n, CBLAS_INT k,
             const double *alpha,
             const double *A, CBLAS_INT ldA,
             const double *B, CBLAS_INT ldB,
             double beta,
             double *C, CBLAS_INT ldC);

// symm
void
cblas_ssymm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            CBLAS_INT m, CBLAS_INT n,
            float alpha,
            const float *A, CBLAS_INT ldA,
            const float *B, CBLAS_INT ldB,
            float beta,
            float *C, CBLAS_INT ldC);


void
cblas_dsymm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            CBLAS_INT m, CBLAS_INT n,
            double alpha,
            const double *A, CBLAS_INT ldA,
            const double *B, CBLAS_INT ldB,
            double beta,
            double *C, CBLAS_INT ldC);

void
cblas_csymm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            CBLAS_INT m, CBLAS_INT n,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *B, CBLAS_INT ldB,
            const float *beta,
            float *C, CBLAS_INT ldC);

void
cblas_zsymm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *B, CBLAS_INT ldB,
            const double *beta,
            double *C, CBLAS_INT ldC);

// syrk
void
cblas_ssyrk(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE trans,
            CBLAS_INT n, CBLAS_INT k,
            float alpha,
            const float *A, CBLAS_INT ldA,
            float beta,
            float *C, CBLAS_INT ldC);

void
cblas_dsyrk(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE trans,
            CBLAS_INT n, CBLAS_INT k,
            double alpha,
            const double *A, CBLAS_INT ldA,
            double beta,
            double *C, CBLAS_INT ldC);

void
cblas_csyrk(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE trans,
            CBLAS_INT n, CBLAS_INT k,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            const float *beta,
            float *C, CBLAS_INT ldC);

void
cblas_zsyrk(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE trans,
            CBLAS_INT n, CBLAS_INT k,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            const double *beta,
            double *C, CBLAS_INT ldC);

// syr2k
void
cblas_ssyr2k(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
             enum CBLAS_TRANSPOSE trans,
            CBLAS_INT n, CBLAS_INT k,
            float alpha,
            const float *A, CBLAS_INT ldA,
            const float *B, CBLAS_INT ldB,
            float beta,
            float *C, CBLAS_INT ldC);

void
cblas_dsyr2k(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
             enum CBLAS_TRANSPOSE trans,
             CBLAS_INT n, CBLAS_INT k,
             double alpha,
             const double *A, CBLAS_INT ldA,
             const double *B, CBLAS_INT ldB,
             double beta,
             double *C, CBLAS_INT ldC);

void
cblas_csyr2k(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
             enum CBLAS_TRANSPOSE trans,
             CBLAS_INT n, CBLAS_INT k,
             const float *alpha,
             const float *A, CBLAS_INT ldA,
             const float *B, CBLAS_INT ldB,
             const float *beta,
             float *C, CBLAS_INT ldC);

void
cblas_zsyr2k(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
             enum CBLAS_TRANSPOSE trans,
             CBLAS_INT n, CBLAS_INT k,
             const double *alpha,
             const double *A, CBLAS_INT ldA,
             const double *B, CBLAS_INT ldB,
             const double *beta,
             double *C, CBLAS_INT ldC);

// trmm
void
cblas_strmm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            float alpha,
            const float *A, CBLAS_INT ldA,
            float *B, CBLAS_INT ldB);

void
cblas_dtrmm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            double alpha,
            const double *A, CBLAS_INT ldA,
            double *B, CBLAS_INT ldB);

void
cblas_ctrmm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            float *B, CBLAS_INT ldB);

void
cblas_ztrmm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            double *B, CBLAS_INT ldB);

// trsm
void
cblas_strsm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            float alpha,
            const float *A, CBLAS_INT ldA,
            float *B, CBLAS_INT ldB);

void
cblas_dtrsm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            double alpha,
            const double *A, CBLAS_INT ldA,
            double *B, CBLAS_INT ldB);

void
cblas_ctrsm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            const float *alpha,
            const float *A, CBLAS_INT ldA,
            float *B, CBLAS_INT ldB);

void
cblas_ztrsm(enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO upLo,
            enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
            CBLAS_INT m, CBLAS_INT n,
            const double *alpha,
            const double *A, CBLAS_INT ldA,
            double *B, CBLAS_INT ldB);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CXXBLAS_DRIVERS_CBLAS_H

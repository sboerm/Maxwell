
#include "linalg.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>

matrix *
new_matrix(int rows, int cols)
{
  matrix *a;

  a = (matrix *) malloc(sizeof(matrix));
  a->rows = rows;
  a->cols = cols;
  a->ld = rows;
  a->a = (field *) malloc(sizeof(field) * rows * cols);

  return a;
}

void
del_matrix(matrix *a)
{
  free(a->a);
  a->a = 0;
  free(a);
}

#ifndef FIELD_COMPLEX
extern void
dpotrf_(const char *uplo,
	const int *n, double *a, const int *lda,
	int *info);

int
potrf_matrix(matrix *a)
{
  int info;

  assert(a->rows == a->cols);
  
  dpotrf_("Lower", &a->rows, a->a, &a->ld, &info);

  return info;
}
#else
extern void
zpotrf_(const char *uplo,
	const int *n, double complex *a, const int *lda,
	int *info);

int
potrf_matrix(matrix *a)
{
  int info;

  assert(a->rows == a->cols);
  
  zpotrf_("Lower", &a->rows, a->a, &a->ld, &info);

  return info;
}
#endif

#ifndef FIELD_COMPLEX
extern void
dtrsm_(const char *side, const char *uplo,
       const char *transa, const char *diag,
       const int *m, const int *n, const double *alpha,
       const double *a, const int *lda,
       double *b, const int *ldb);

void
potrsv_matrix(const matrix *a, field *b)
{
  const field one = 1.0;
  const ione = 1;
  
  assert(a->rows == a->cols);

  dtrsm_("Left", "Lower", "Not transposed", "Not unit triangular",
	 &a->rows, &ione, &one, a->a, &a->ld,
	 b, &a->rows);

  dtrsm_("Left", "Lower", "Transposed", "Not unit triangular",
	 &a->rows, &ione, &one, a->a, &a->ld,
	 b, &a->rows);
}
#else
extern void
ztrsm_(const char *side, const char *uplo,
       const char *transa, const char *diag,
       const int *m, const int *n, const double complex *alpha,
       const double complex *a, const int *lda,
       double complex *b, const int *ldb);

void
potrsv_matrix(const matrix *a, field *b)
{
  const field one = 1.0;
  const int ione = 1;
  
  assert(a->rows == a->cols);

  ztrsm_("Left", "Lower", "Not transposed", "Not unit triangular",
	 &a->rows, &ione, &one, a->a, &a->ld,
	 b, &a->rows);

  ztrsm_("Left", "Lower", "Conjugated", "Not unit triangular",
	 &a->rows, &ione, &one, a->a, &a->ld,
	 b, &a->rows);
}
#endif

#ifndef FIELD_COMPLEX
extern void
dsyev_(const char *jobz, const char *uplo,
       const int *n, double *a, const int *lda,
       double *w, double *work, const int *lwork, int *info);

int
syev_matrix(matrix *a, real *lambda)
{
  field *work;
  int lwork, info;

  assert(a->rows == a->cols);

  if(a->rows == 0)
    return 0;

  lwork = 10 * a->rows;
  work = (field *) malloc(sizeof(field) * lwork);
  
  dsyev_("V", "L", &a->rows, a->a, &a->ld,
	 lambda, work, &lwork, &info);
  
  free(work);
  
  return info;
}

extern void
dsygv_(const int *itype, const char *jobz, const char *uplo,
       const int *n, double *a, const int *lda,
       double *b, const int *ldb, double *w, double *work,
       int *lwork, int *info);

int
sygv_matrix(matrix *a, matrix *b, real *lambda)
{
  const int itype = 1;
  field *work;
  int lwork, info;

  assert(a->rows == a->cols);
  assert(b->rows == b->cols);
  assert(a->rows == b->rows);

  if(a->rows == 0)
    return 0;

  lwork = 10 * a->rows;
  work = (field *) malloc(sizeof(field) * lwork);
  
  dsygv_(&itype, "V", "L", &a->rows, a->a, &a->ld, b->a, &b->ld,
	 lambda, work, &lwork, &info);
  
  free(work);
  
  return info;
}

void
sygv2_matrix(matrix *a, matrix *b, real *lambda)
{
  field mu, delta;
  field x, y;

  mu = 0.5 * (a->a[3] * b->a[0] + a->a[0] * b->a[3] - 2.0 * a->a[1] * b->a[1]) / (b->a[0] * b->a[3] - b->a[1] * b->a[1]);
  
  delta = mu * mu - (a->a[0] * a->a[3] - a->a[1] * a->a[1]) / (b->a[0] * b->a[3] - b->a[1] * b->a[1]);

  lambda[1] = mu + sqrt(delta);
  lambda[0] = (mu * mu - delta) / lambda[1];

  x = 1.0;
  y = (a->a[0] - lambda[0] * b->a[0]) / (lambda[0] * b->a[1] - a->a[1]);
  delta = sqrt(x * (b->a[0] * x + b->a[1] * y) + y * (b->a[1] * x + b->a[3] * y));
  x /= delta;
  y /= delta;

  a->a[0] = x;
  a->a[1] = y;

  x = 1.0;
  y = (a->a[0] - lambda[1] * b->a[0]) / (lambda[1] * b->a[1] - a->a[1]);
  delta = sqrt(x * (b->a[0] * x + b->a[1] * y) + y * (b->a[1] * x + b->a[3] * y));
  x /= delta;
  y /= delta;

  a->a[2] = x;
  a->a[3] = y;
}

extern void
dcopy_(const int *n, const double *dx, const int *incx,
       double *dy, const int *incy);

extern void
dggev_(const char *jobvl, const char *jobvr,
       const int *n,
       double *a, const int *lda,
       double *b, const int *ldb,
       double *alphar, double *alphai, double *beta,
       double *vl, const int *ldvl, double *vr, const int *ldvr,
       double *work, const int *lwork, int *info);

int
ggev_matrix(matrix *a, matrix *b,
	    real *alphar, real *alphai, real *beta)
{
  const int one = 1;
  field *vl;
  field *work;
  int lwork, info;
  int j;

  assert(a->rows == a->cols);
  assert(b->rows == b->cols);
  assert(a->rows == b->rows);

  if(a->rows == 0)
    return 0;

  vl = (field *) malloc(sizeof(field) * a->rows * a->rows);
  lwork = 20 * a->rows;
  work = (field *) malloc(sizeof(field) * lwork);

  dggev_("V", "N", &a->rows, a->a, &a->ld, b->a, &b->ld,
	 alphar, alphai, beta, vl, &a->rows, 0, &one,
	 work, &lwork, &info);

  for(j=0; j<a->cols; j++)
    dcopy_(&a->rows, vl+j*a->rows, &one, a->a+j*a->ld, &one);

  free(work);
  free(vl);
  
  return info;
}
#else
extern void
zheev_(const char *jobz, const char *uplo,
       const int *n, complex double *a, const int *lda,
       double *w, complex double *work, const int *lwork,
       double *rwork, int *info);

int
syev_matrix(matrix *a, real *lambda)
{
  field *work;
  real *rwork;
  int lwork, info;

  assert(a->rows == a->cols);

  if(a->rows == 0)
    return 0;

  lwork = 10 * a->rows;
  work = (field *) malloc(sizeof(field) * lwork);
  rwork = (real *) malloc(sizeof(real) * 3 * a->rows);
  
  zheev_("V", "L", &a->rows, a->a, &a->ld,
	 lambda, work, &lwork, rwork, &info);

  free(rwork);
  free(work);
  
  return info;
}

extern void
zhegv_(const int *itype, const char *jobz, const char *uplo,
       const int *n, complex double *a, const int *lda,
       complex double *b, const int *ldb, double *w, complex double *work,
       int *lwork, double *rwork, int *info);

int
sygv_matrix(matrix *a, matrix *b, real *lambda)
{
  const int itype = 1;
  field *work;
  real *rwork;
  int lwork, info;

  assert(a->rows == a->cols);
  assert(b->rows == b->cols);
  assert(a->rows == b->rows);

  if(a->rows == 0)
    return 0;

  lwork = 10 * a->rows;
  work = (field *) malloc(sizeof(field) * lwork);
  rwork = (real *) malloc(sizeof(real) * 3 * a->rows);
  
  zhegv_(&itype, "V", "L", &a->rows, a->a, &a->ld, b->a, &b->ld,
	 lambda, work, &lwork, rwork, &info);

  free(rwork);
  free(work);
  
  return info;
}
#endif

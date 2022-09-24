
#ifndef LINALG_H
#define LINALG_H

#include "settings.h"

typedef struct {
  int rows;
  int cols;
  int ld;

  field *a;
} matrix;

matrix *
new_matrix(int rows, int cols);

void
del_matrix(matrix *a);

int
potrf_matrix(matrix *a);

void
potrsv_matrix(const matrix *a, field *b);

int
syev_matrix(matrix *a, real *lambda);

int
sygv_matrix(matrix *a, matrix *b, real *lambda);

/*
void
sygv2_matrix(matrix *a, matrix *b, real *lambda);

int
ggev_matrix(matrix *a, matrix *b,
	    real *alphar, real *alphai, real *beta);
*/

#endif

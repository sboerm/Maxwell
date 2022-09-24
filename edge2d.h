
#ifndef EDGE2D_H
#define EDGE2D_H

#include "grid2d.h"
#include <stdlib.h>

#ifdef USE_CAIRO
#include "cairo.h"
#endif

typedef struct {
  /** @brief Grid */
  const grid2d *gr;

  /** @brief Coefficients for x-basis functions.
   *
   *  This is an array with <tt>gr->ny+1</tt> rows containing
   *  <tt>gr->nx</tt> x-edges. The j-th row starts at
   *  <tt>x + j*nx</tt>. */
  field *x;

  /** @brief Coefficients for y-basis functions.
   *
   *  This is an array with <tt>gr->ny</tt> rows containing
   *  <tt>gr->nx+1</tt> y-edges. The j-th row starts at
   *  <tt>y + j*(nx+1)</tt>. */
  field *y;
} edge2d;

#include "node2d.h"
#include "linalg.h"
#include <stdbool.h>

#ifdef USE_NETCDF
#include <netcdf.h>

typedef struct {
  int file;

  int nc_blochsteps;
  int blochsteps;

  int nc_eigenvalues;
  int eigenvalues;

  int nc_graphx;
  int graphx;

  int nc_graphy;
  int graphy;

  int nc_lambda;
  int nc_vectorxr;
  int nc_vectorxi;
  int nc_vectoryr;
  int nc_vectoryi;
  int nc_xbloch;
  int nc_ybloch;
  int nc_xfactorr;
  int nc_xfactori;
  int nc_yfactorr;
  int nc_yfactori;
} ncfile;
#endif

/* ----------------------------------------
 * Constructor and destructor
 * ---------------------------------------- */

edge2d *
new_edge2d(const grid2d *gr);

void
del_edge2d(edge2d *x);

size_t
getdimension_edge2d(const edge2d *x);

size_t
getsize_edge2d(const edge2d *x);

/* ----------------------------------------
 * Basic utility functions
 * ---------------------------------------- */

void
zero_edge2d(edge2d *x);

void
copy_edge2d(const edge2d *x, edge2d *y);

void
swap_edge2d(edge2d *x, edge2d *y);

void
random_edge2d(edge2d *x);

#ifdef USE_CAIRO
void
cairodraw_edge2d(const edge2d *x, int gx, int gy,
		 const epspattern *pat, const char *filename);
#endif

#ifdef USE_NETCDF
ncfile *
new_ncfile(const edge2d *x, int eigs, int blochsteps, int gx, int gy,
	   const epspattern *pat, const char *filename);

void
del_ncfile(ncfile *nf);

void
write_ncfile(ncfile *nf, int blochstep, const real *lambda,
	     const edge2d **eigs);
#endif

/* ----------------------------------------
 * Discretization
 * ---------------------------------------- */

void
interpolate_edge2d(edge2d *x, void (*func)(const real *x, field *fx, void *data), void *data);

void
l2functional_edge2d(edge2d *x, edge2d *xbuf, edge2d *ybuf,
		    void (*func)(const real *x, field *fx, void *data),
		    void *data);

/* ----------------------------------------
 * Basic linear algebra
 * ---------------------------------------- */

void
scale_edge2d(field alpha, edge2d *x);

void
add_edge2d(field alpha, const edge2d *x, edge2d *y);

real
norm2_edge2d(const edge2d *x);

real
normmax_edge2d(const edge2d *x);

real
l2norm_edge2d(const edge2d *x,
	      void (*func)(const real *x, field *fx, void *data),
	      void *data);

field
dotprod_edge2d(const edge2d *x, const edge2d *y);

void
nullprod_edge2d(const edge2d *x, field *xprod, field *yprod);

void
nulladd_edge2d(field xalpha, field yalpha, edge2d *x);

void
center_edge2d(edge2d *x);

/* ----------------------------------------
 * Matrix-vector multiplication with the system matrix
 * ---------------------------------------- */

void
addeval_edge2d(field alpha, field beta,
	       const edge2d *x, edge2d *y);

field
energyprod_edge2d(field alpha, field beta,
		  const edge2d *x, const edge2d *y);

field
massprod_edge2d(const edge2d *x, const edge2d *y);

/* ----------------------------------------
 * Gauss-Seidel iteration
 * ---------------------------------------- */

void
gsforward_edge2d(field alpha, field beta,
		 const edge2d *b, edge2d *x);

void
gsbackward_edge2d(field alpha, field beta,
		  const edge2d *b, edge2d *x);

void
gssymm_edge2d(field alpha, field beta,
	      const edge2d *b, edge2d *x);

void
gsforward_simple_edge2d(field alpha, field beta,
			const edge2d *b, edge2d *x);

void
gsbackward_simple_edge2d(field alpha, field beta,
			 const edge2d *b, edge2d *x);

void
gssymm_simple_edge2d(field alpha, field beta,
		     const edge2d *b, edge2d *x);

/* ----------------------------------------
 * Intergrid transfer
 * ---------------------------------------- */

void
prolongation_edge2d(field alpha, const edge2d *c, edge2d *f);

void
restriction_edge2d(field alpha, const edge2d *f, edge2d *c);

/* ----------------------------------------
 * Multigrid iteration
 * ---------------------------------------- */

void
vcycle_edge2d(int L, int nu, const matrix *Ac,
	      edge2d **b, edge2d **x, edge2d **d);

void
hcycle_edge2d(int L, int nu, const matrix *Ac,
	      edge2d **b, edge2d **x, edge2d **d,
	      node2d **bg, node2d **xg);

/* ----------------------------------------
 * CG with multigrid preconditioner
 * ---------------------------------------- */

void
mgcginit_edge2d(int L, int nu, const edge2d *b, edge2d *x,
		edge2d *p, edge2d *a,
		edge2d **r, edge2d **q, edge2d **d);

void
mgcgstep_edge2d(int L, int nu, const edge2d *b, edge2d *x,
		edge2d *p, edge2d *a,
		edge2d **r, edge2d **q, edge2d **d);

/* ----------------------------------------
 * Dense matrix algebra
 * ---------------------------------------- */

void
densematrix_edge2d(field alpha, field beta,
		   const grid2d *gr, matrix *a);

void
denseto_edge2d(const field *x, edge2d *y);

void
densefrom_edge2d(const edge2d *x, field *y);

/* ----------------------------------------
 * Householder factorization
 * ---------------------------------------- */

void
unitindices_edge2d(int idx, const edge2d *x, int *i, int *j);

void
unit_edge2d(int idx, edge2d *x);

field
massprod_xunit_edge2d(int i, int j, const edge2d *y);

void
add_xunit_edge2d(field alpha, int i, int j, edge2d *y);

void
buildhouseholder_edge2d(int idx, edge2d *x, field *tau);

void
applyhouseholder_edge2d(const edge2d *v, field tau, edge2d *x);

void
orthonormalize_edge2d(int k, edge2d **x, field *tau, edge2d *v);

/* ------------------------------------------------------------
 * Preconditioned inverse iteration with gradient elimination
 * ------------------------------------------------------------ */

void
pinvit_edge2d(int l,
	      int smoother_steps, int prec_steps, int gradient_steps,
	      real lambda, edge2d *e,
	      const matrix *Ae, edge2d **b, edge2d **x, edge2d **d,
	      const matrix *An, node2d **bg, node2d **xg, node2d **dg);

#endif

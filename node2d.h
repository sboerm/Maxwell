
#ifndef NODE2D_H
#define NODE2D_H

#include "grid2d.h"

#include <stdbool.h>

typedef struct {
  /** @brief Grid */
  const grid2d *gr;

  /** @brief Coefficients for nodal basis functions.
   *
   *  This is an array with <tt>gr->ny+1</tt> rows containing
   *  <tt>gr->nx+1</tt> nodal values. The j-th row starts at
   *  <tt>x + j*(nx+1)</tt>. */
  field *v;
} node2d;

#include "edge2d.h"
#include "linalg.h"

/* ----------------------------------------
 * Constructor and destructor
 * ---------------------------------------- */

node2d *
new_node2d(const grid2d *gr);

void
del_node2d(node2d *x);

/* ----------------------------------------
 * Basic utility functions
 * ---------------------------------------- */

void
zero_node2d(node2d *x);

void
copy_node2d(const node2d *x, node2d *y);

void
random_node2d(node2d *x);

/* ----------------------------------------
 * Discretization
 * ---------------------------------------- */

void
interpolate_node2d(node2d *x, field (*func)(const real *x, void *data), void *data);

void
l2functional_node2d(node2d *x, node2d *buf,
		    field (*func)(const real *x, void *data),
		    void *data);

real
l2norm_node2d(const node2d *x,
	      field (*func)(const real *x, void *data),
	      void *data);

/* ----------------------------------------
 * Basic linear algebra
 * ---------------------------------------- */

void
scale_node2d(field alpha, node2d *x);

void
add_node2d(field alpha, const node2d *x, node2d *y);

real
norm2_node2d(const node2d *x);

real
normmax_node2d(const node2d *x);

field
dotprod_node2d(const node2d *x, const node2d *y);

void
center_node2d(node2d *x);

/* ----------------------------------------
 * Matrix-vector multiplication with the system matrix
 * ---------------------------------------- */

void
addeval_node2d(field alpha, const node2d *x, node2d *y);

/* ----------------------------------------
 * Gauss-Seidel iteration
 * ---------------------------------------- */

void
gsforward_node2d(field alpha,
		 const node2d *b, node2d *x);

void
gsbackward_node2d(field alpha,
		  const node2d *b, node2d *x);

void
gssymm_node2d(field alpha,
	      const node2d *b, node2d *x);

/* ----------------------------------------
 * Intergrid transfer
 * ---------------------------------------- */

void
prolongation_node2d(field alpha, const node2d *c, node2d *f);

void
restriction_node2d(field alpha, const node2d *f, node2d *c);

void
gradient_node2d(field alpha, const node2d *x, edge2d *y);

void
adjgradient_node2d(field alpha, const edge2d *x, node2d *y);

/* ----------------------------------------
 * Multigrid iteration
 * ---------------------------------------- */

void
vcycle_node2d(int L, int nu, const matrix *Ac,
	      node2d **b, node2d **x, node2d **d);

/* ----------------------------------------
 * CG with multigrid preconditioner
 * ---------------------------------------- */

void
mgcginit_node2d(int L, int nu, const node2d *b, node2d *x,
		node2d *p, node2d *a,
		node2d **r, node2d **q, node2d **d);

void
mgcgstep_node2d(int L, int nu, const node2d *b, node2d *x,
		node2d *p, node2d *a,
		node2d **r, node2d **q, node2d **d);

/* ----------------------------------------
 * Dense matrix algebra
 * ---------------------------------------- */

void
densematrix_node2d(bool fixnull, const grid2d *gr, matrix *a);

void
denseto_node2d(const field *x, node2d *y);

void
densefrom_node2d(const node2d *x, field *y);

#endif

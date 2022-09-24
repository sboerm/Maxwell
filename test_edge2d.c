
#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "edge2d.h"
#include "basic.h"

static void
example1_sol(const real *x, field *fx, void *data)
{
  (void) data;

  fx[0] = cos(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
  fx[1] = sin(2.0 * M_PI * x[0]) * cos(2.0 * M_PI * x[1]);
}

static void
example1_rhs(const real *x, field *fx, void *data)
{
  (void) data;

  fx[0] = cos(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
  fx[1] = sin(2.0 * M_PI * x[0]) * cos(2.0 * M_PI * x[1]);
}

const field c11 = 2.0 * M_PI * I;
const field c12 = 4.0 * M_PI * I;
const field c21 = 4.0 * M_PI * I;
const field c22 = 8.0 * M_PI * I;

static void
example2_sol(const real *x, field *fx, void *data)
{
  (void) data;

  fx[0] = EXP(c11 * x[0]) * EXP(c12 * x[1]);
  fx[1] = EXP(c21 * x[0]) * EXP(c22 * x[1]);
}

static void
example2_rhs(const real *x, field *fx, void *data)
{
  (void) data;

  fx[0] = c21 * c22 * EXP(c21 * x[0]) * EXP(c22 * x[1])
    - c12 * c12 * EXP(c11 * x[0]) * EXP(c12 * x[1])
    + EXP(c11 * x[0]) * EXP(c12 * x[1]);
  fx[1] = c12 * c11 * EXP(c11 * x[0]) * EXP(c12 * x[1])
    - c21 * c21 * EXP(c21 * x[0]) * EXP(c22 * x[1])
    + EXP(c21 * x[0]) * EXP(c22 * x[1]);
}

int
main(int argc, char **argv)
{
  grid2d *gr, *grf;
  edge2d *x, *b, *d, *xf, *bf, *xm[4];
  real oldnorm, norm, error;
  field product1, product2, tau, taum[4];
  matrix *A;
  field *xd, *bd;
  int nx, ny;
  real t_run;
  int i, j;

  nx = 32;

  if(argc > 1)
    sscanf(argv[1], "%d", &nx);

  ny = nx;

  if(argc > 2)
    sscanf(argv[2], "%d", &ny);

#pragma omp parallel private(i,oldnorm,norm)
  {
#pragma omp single
    {
      printf("========================================\n"
	     "Grid function tests, %d x %d boxes\n"
	     "========================================\n",
	     nx, ny);
      
      gr = new_grid2d(nx, ny, 1.0/nx, 1.0/ny);

      gr->xfactor = I;

      x = new_edge2d(gr);
      b = new_edge2d(gr);
      d = new_edge2d(gr);

      xm[0] = new_edge2d(gr);
      xm[1] = new_edge2d(gr);
      xm[2] = new_edge2d(gr);
      xm[3] = new_edge2d(gr);

      grf = new_grid2d(2*nx, 2*ny, 0.5*gr->hx, 0.5*gr->hy);
      grf->xfactor = gr->xfactor;
      grf->yfactor = gr->yfactor;
      
      xf = new_edge2d(grf);
      bf = new_edge2d(grf);
    }

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing first solution\n"
	 "----------------------------------------\n");
#pragma omp single
  printf("Interpolating solution\n");
  interpolate_edge2d(x, example1_sol, 0);

  error = l2norm_edge2d(x, example1_sol, 0);
#pragma omp single
  printf("  L^2 error %.4e\n", error);

#pragma omp single
  printf("Evaluating differential operator\n");
  zero_edge2d(b);
  addeval_edge2d(1.0, 0.0, x, b);
  error = norm2_edge2d(b);
#pragma omp single
  printf("  Error %.4e\n", error);

#pragma omp single
  printf("Integrating right-hand side\n");
  l2functional_edge2d(b, x, d, example1_rhs, 0);
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Interpolating solution\n");
  interpolate_edge2d(x, example1_sol, 0);

  error = l2norm_edge2d(x, example1_sol, 0);
#pragma omp single
  printf("  L^2 error %.4e\n", error);

#pragma omp single
  printf("Evaluating mass operator\n");
  addeval_edge2d(0.0, -1.0, x, b);
  error = norm2_edge2d(b);
#pragma omp single
  printf("  Error %.4e (%.4e)\n", error, error/norm);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing second solution\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Integrating right-hand side\n");
  l2functional_edge2d(b, x, d, example2_rhs, 0);
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Interpolating solution\n");
  interpolate_edge2d(x, example2_sol, 0);

  error = l2norm_edge2d(x, example2_sol, 0);
#pragma omp single
  printf("  L^2 error %.4e\n", error);

#pragma omp single
  printf("Evaluating operator\n");
  addeval_edge2d(-1.0, -1.0, x, b);
  error = norm2_edge2d(b);
#pragma omp single
  printf("  Error %.4e (%.4e)\n", error, error/norm);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing self-adjointness\n"
	 "----------------------------------------\n");

  random_edge2d(b);
  random_edge2d(d);
  zero_edge2d(x);
  addeval_edge2d(1.0, 1.0, b, x);
  product1 = dotprod_edge2d(d, x);
  oldnorm = norm2_edge2d(d) * norm2_edge2d(x);
  zero_edge2d(x);
  addeval_edge2d(1.0, 1.0, d, x);
  product2 = dotprod_edge2d(x, b);
  norm = norm2_edge2d(x) * norm2_edge2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing energy product\n"
	 "----------------------------------------\n");

  zero_edge2d(x);
  addeval_edge2d(1.0, 1.0, b, x);
  product1 = dotprod_edge2d(d, x);
  norm = norm2_edge2d(d) * norm2_edge2d(x);
  product2 = energyprod_edge2d(1.0, 1.0, d, b);
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing forward GS\n"
	 "----------------------------------------\n");

  zero_edge2d(b);
  random_edge2d(x);
  
  copy_edge2d(b, d);
  addeval_edge2d(-1.0, -1.0, x, d);
  norm = norm2_edge2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  t_run = omp_get_wtime();
  for(i=0; i<10; i++) {
    gsforward_edge2d(1.0, 1.0, b, x);
    
    copy_edge2d(b, d);
    addeval_edge2d(-1.0, -1.0, x, d);
    oldnorm = norm;
    norm = norm2_edge2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  t_run = reduce_max_real(omp_get_wtime() - t_run);
#pragma omp single
  printf("  %.1f seconds (%.1f milliseconds per iteration)\n",
	 t_run, 1000.0 * t_run / 10);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing backward GS\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Testing symmetry\n");
  random_edge2d(b);
  random_edge2d(d);
  zero_edge2d(x);
  gsforward_edge2d(1.0, 1.0, b, x);
  product1 = dotprod_edge2d(d, x);
  oldnorm = norm2_edge2d(d) * norm2_edge2d(x);
  zero_edge2d(x);
  gsbackward_edge2d(1.0, 1.0, d, x);
  product2 = dotprod_edge2d(x, b);
  norm = norm2_edge2d(x) * norm2_edge2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);
  
  zero_edge2d(b);
  random_edge2d(x);

  copy_edge2d(b, d);
  addeval_edge2d(-1.0, -1.0, x, d);
  norm = norm2_edge2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  t_run = omp_get_wtime();
  for(i=0; i<10; i++) {
    gsbackward_edge2d(1.0, 1.0, b, x);
    
    copy_edge2d(b, d);
    addeval_edge2d(-1.0, -1.0, x, d);
    oldnorm = norm;
    norm = norm2_edge2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  t_run = reduce_max_real(omp_get_wtime() - t_run);
#pragma omp single
  printf("  %.1f seconds (%.1f milliseconds per iteration)\n",
	 t_run, 1000.0 * t_run / 10);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing symmetric GS\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Testing symmetry\n");
  random_edge2d(b);
  random_edge2d(d);
  zero_edge2d(x);
  gssymm_edge2d(1.0, 1.0, b, x);
  product1 = dotprod_edge2d(d, x);
  oldnorm = norm2_edge2d(d) * norm2_edge2d(x);
  zero_edge2d(x);
  gssymm_edge2d(1.0, 1.0, d, x);
  product2 = dotprod_edge2d(x, b);
  norm = norm2_edge2d(x) * norm2_edge2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);

  zero_edge2d(b);
  random_edge2d(x);

  copy_edge2d(b, d);
  addeval_edge2d(-1.0, -1.0, x, d);
  norm = norm2_edge2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  t_run = omp_get_wtime();
  for(i=0; i<10; i++) {
    gssymm_edge2d(1.0, 1.0, b, x);
    
    copy_edge2d(b, d);
    addeval_edge2d(-1.0, -1.0, x, d);
    oldnorm = norm;
    norm = norm2_edge2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  t_run = reduce_max_real(omp_get_wtime() - t_run);
#pragma omp single
  printf("  %.1f seconds (%.1f milliseconds per iteration)\n",
	 t_run, 1000.0 * t_run / 10);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing simple forward GS\n"
	 "----------------------------------------\n");

  zero_edge2d(b);
  random_edge2d(x);

  copy_edge2d(b, d);
  addeval_edge2d(-1.0, -1.0, x, d);
  norm = norm2_edge2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  t_run = omp_get_wtime();
  for(i=0; i<10; i++) {
    gsforward_simple_edge2d(1.0, 1.0, b, x);
    
    copy_edge2d(b, d);
    addeval_edge2d(-1.0, -1.0, x, d);
    oldnorm = norm;
    norm = norm2_edge2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  t_run = reduce_max_real(omp_get_wtime() - t_run);
#pragma omp single
  printf("  %.1f seconds (%.1f milliseconds per iteration)\n",
	 t_run, 1000.0 * t_run / 10);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing simple backward GS\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Testing symmetry\n");
  random_edge2d(b);
  random_edge2d(d);
  zero_edge2d(x);
  gsforward_simple_edge2d(1.0, 1.0, b, x);
  product1 = dotprod_edge2d(d, x);
  oldnorm = norm2_edge2d(d) * norm2_edge2d(x);
  zero_edge2d(x);
  gsbackward_simple_edge2d(1.0, 1.0, d, x);
  product2 = dotprod_edge2d(x, b);
  norm = norm2_edge2d(x) * norm2_edge2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2),
	 ABS(product1-product2)/norm);

  zero_edge2d(b);
  random_edge2d(x);

  t_run = omp_get_wtime();
  copy_edge2d(b, d);
  addeval_edge2d(-1.0, -1.0, x, d);
  norm = norm2_edge2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  for(i=0; i<10; i++) {
    gsbackward_simple_edge2d(1.0, 1.0, b, x);
    
    copy_edge2d(b, d);
    addeval_edge2d(-1.0, -1.0, x, d);
    oldnorm = norm;
    norm = norm2_edge2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  t_run = reduce_max_real(omp_get_wtime() - t_run);
#pragma omp single
  printf("  %.1f seconds (%.1f milliseconds per iteration)\n",
	 t_run, 1000.0 * t_run / 10);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing simple symmetric GS\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Testing symmetry\n");
  random_edge2d(b);
  random_edge2d(d);
  zero_edge2d(x);
  gssymm_simple_edge2d(1.0, 1.0, b, x);
  product1 = dotprod_edge2d(d, x);
  oldnorm = norm2_edge2d(d) * norm2_edge2d(x);
  zero_edge2d(x);
  gssymm_simple_edge2d(1.0, 1.0, d, x);
  product2 = dotprod_edge2d(x, b);
  norm = norm2_edge2d(x) * norm2_edge2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);

  zero_edge2d(b);
  random_edge2d(x);

  t_run = omp_get_wtime();
  copy_edge2d(b, d);
  addeval_edge2d(-1.0, -1.0, x, d);
  norm = norm2_edge2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  for(i=0; i<10; i++) {
    gssymm_simple_edge2d(1.0, 1.0, b, x);
    
    copy_edge2d(b, d);
    addeval_edge2d(-1.0, -1.0, x, d);
    oldnorm = norm;
    norm = norm2_edge2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  t_run = reduce_max_real(omp_get_wtime() - t_run);
#pragma omp single
  printf("  %.1f seconds (%.1f milliseconds per iteration)\n",
	 t_run, 1000.0 * t_run / 10);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing prolongation\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vector\n");
  random_edge2d(x);
  norm = norm2_edge2d(x);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Applying prolongation\n");
  zero_edge2d(xf);
  prolongation_edge2d(1.0, x, xf);
  norm = norm2_edge2d(xf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Multiplying by the system matrix on the fine grid\n");
  zero_edge2d(bf);
  addeval_edge2d(1.0, 1.0, xf, bf);
  norm = norm2_edge2d(bf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Multiplying by the system matrix on the coarse grid\n");
  zero_edge2d(b);
  addeval_edge2d(1.0, 1.0, x, b);
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Computing inner products\n");
  product1 = dotprod_edge2d(x, b);
  product2 = dotprod_edge2d(xf, bf);
#pragma omp single
  printf("  Difference %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/ABS(product1));

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing restriction\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vectors\n");
  random_edge2d(x);
  random_edge2d(bf);
  norm = norm2_edge2d(x);
  oldnorm = norm2_edge2d(bf);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);

#pragma omp single
  printf("Applying prolongation and restriction\n");
  zero_edge2d(xf);
  prolongation_edge2d(1.0, x, xf);
  zero_edge2d(b);
  restriction_edge2d(1.0, bf, b);
  norm = norm2_edge2d(xf);
  oldnorm = norm2_edge2d(b);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);

#pragma omp single
  printf("Computing inner products\n");
  product1 = dotprod_edge2d(x, b);
  product2 = dotprod_edge2d(xf, bf);
#pragma omp single
  printf("  Difference %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/ABS(product1));

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing Galerkin property\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vector\n");
  random_edge2d(x);
  norm = norm2_edge2d(x);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Applying prolongation\n");
  zero_edge2d(xf);
  prolongation_edge2d(1.0, x, xf);
  norm = norm2_edge2d(xf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Multiplying by the system matrix on the fine grid\n");
  zero_edge2d(bf);
  addeval_edge2d(1.0, 0.0, xf, bf);
  norm = norm2_edge2d(bf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Applying restriction\n");
  zero_edge2d(b);
  restriction_edge2d(1.0, bf, b);
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Subtracting coarse matrix-vector product\n");
  addeval_edge2d(-1.0, 0.0, x, b);
  oldnorm = norm;
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e (%.4e)\n", norm, norm/oldnorm);

#pragma omp single
  printf("Applying prolongation\n");
  zero_edge2d(xf);
  prolongation_edge2d(1.0, x, xf);
  norm = norm2_edge2d(xf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

  #pragma omp single
  printf("Multiplying by the mass matrix on the fine grid\n");
  zero_edge2d(bf);
  addeval_edge2d(0.0, 1.0, xf, bf);
  norm = norm2_edge2d(bf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Applying restriction\n");
  zero_edge2d(b);
  restriction_edge2d(1.0, bf, b);
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Subtracting coarse matrix-vector product\n");
  addeval_edge2d(0.0, -1.0, x, b);
  oldnorm = norm;
  norm = norm2_edge2d(b);
#pragma omp single
  printf("  Norm %.4e (%.4e)\n", norm, norm/oldnorm);

  if(2*nx*ny <= 8192) {
#pragma omp single
    printf("----------------------------------------\n"
	   "Testing dense matrix conversion\n"
	   "----------------------------------------\n");
#pragma omp single
    {
      A = new_matrix(2*nx*ny, 2*nx*ny);
      xd = (field *) malloc(sizeof(field) * 2*nx*ny);
      bd = (field *) malloc(sizeof(field) * 2*nx*ny);
      densematrix_edge2d(1.0, 1.0, gr, A);
    }
    
    random_edge2d(x);
    
#pragma omp single
    {
      densefrom_edge2d(x, xd);
      for(i=0; i<2*nx*ny; i++)
	bd[i] = 0.0;
      for(j=0; j<2*nx*ny; j++)
	for(i=0; i<2*nx*ny; i++)
	  bd[i] += A->a[i+j*A->ld] * xd[j];
      denseto_edge2d(bd, b);
    }
    norm = norm2_edge2d(b);
    addeval_edge2d(-1.0, -1.0, x, b);
    error = norm2_edge2d(b);

#pragma omp single
    {
      printf("  Error %.2e (%.2e)\n",
	     error, error/norm);
      free(bd);
      free(xd);
      free(A);
    }
  }

#ifdef USE_CAIRO
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing Cairo drawing\n"
	 "----------------------------------------\n");
  interpolate_edge2d(x, example1_sol, 0);
  cairodraw_edge2d(x, 64, 64, 0, "example1.pdf");
  cairodraw_edge2d(x, 64, 64, 0, "example1.png");
#endif
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing orthogonal unit vectors\n"
	 "----------------------------------------\n");
  unit_edge2d(0, x);
  unit_edge2d(1, b);
  unit_edge2d(nx, d);

  norm = massprod_edge2d(x, x);
  product1 = massprod_edge2d(x, b);
  product2 = massprod_edge2d(x, d);
#pragma omp single
  printf("  %.2f (%.2f,%.2f) (%.2f,%.2f)\n", norm, REAL(product1), IMAG(product1), REAL(product2), IMAG(product2));
  product1 = massprod_edge2d(b, x);
  norm = REAL(massprod_edge2d(b, b));
  product2 = massprod_edge2d(b, d);
#pragma omp single
  printf("  (%.2f,%.2f) %.2f (%.2f,%.2f)\n", REAL(product1), IMAG(product1), norm, REAL(product2), IMAG(product2));
  product1 = massprod_edge2d(d, x);
  product2 = massprod_edge2d(d, b);
  norm = REAL(massprod_edge2d(d, d));
#pragma omp single
  printf("  (%.2f,%.2f) (%.2f,%.2f) %.2f\n", REAL(product1), IMAG(product1), REAL(product2), IMAG(product2), norm);
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing Householder reflections\n"
	 "----------------------------------------\n");
  zero_edge2d(x);
  unit_edge2d(0, b);
  add_edge2d(1.0, b, x);
  add_xunit_edge2d(-1.0, 0, 1, x);
  norm = norm2_edge2d(x);
#pragma omp single
  printf("  Unit vector addition error %.2e\n", norm);

  random_edge2d(x);
  unit_edge2d(0, b);
  product1 = massprod_edge2d(b, x);
  product2 = massprod_xunit_edge2d(0, 1, x);
#pragma omp single
  printf("  Unit vector multiplication error %.2e\n",
	 fabs(product1 - product2));
  
  random_edge2d(x);

  copy_edge2d(x, b);
  buildhouseholder_edge2d(0, b, &tau);
  applyhouseholder_edge2d(b, tau, x);

  oldnorm = norm2_edge2d(x);
  unit_edge2d(0, b);
  product1 = massprod_edge2d(b, x);
  add_edge2d(-product1, b, x);

  norm = norm2_edge2d(x);
#pragma omp single
  printf("  %.3e -> %.3e (%.3e)\n", oldnorm, norm, norm/oldnorm);
  }

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing Householder orthonormalization\n"
	 "----------------------------------------\n");
  random_edge2d(xm[0]);
  random_edge2d(xm[1]);
  random_edge2d(xm[2]);
  random_edge2d(xm[3]);

  copy_edge2d(xm[0], b);
  copy_edge2d(xm[1], d);

  orthonormalize_edge2d(4, xm, taum, x);

  taum[0] = massprod_edge2d(xm[0], xm[0]);
  taum[1] = massprod_edge2d(xm[0], xm[1]);
  taum[2] = massprod_edge2d(xm[0], xm[2]);
  taum[3] = massprod_edge2d(xm[0], xm[3]);
#pragma omp single
  printf("  %.2e %.2e %.2e %.2e\n",
	 ABS(taum[0]), ABS(taum[1]), ABS(taum[2]), ABS(taum[3]));

  taum[0] = massprod_edge2d(xm[1], xm[0]);
  taum[1] = massprod_edge2d(xm[1], xm[1]);
  taum[2] = massprod_edge2d(xm[1], xm[2]);
  taum[3] = massprod_edge2d(xm[1], xm[3]);
#pragma omp single
  printf("  %.2e %.2e %.2e %.2e\n",
	 ABS(taum[0]), ABS(taum[1]), ABS(taum[2]), ABS(taum[3]));

  taum[0] = massprod_edge2d(xm[2], xm[0]);
  taum[1] = massprod_edge2d(xm[2], xm[1]);
  taum[2] = massprod_edge2d(xm[2], xm[2]);
  taum[3] = massprod_edge2d(xm[2], xm[3]);
#pragma omp single
  printf("  %.2e %.2e %.2e %.2e\n",
	 ABS(taum[0]), ABS(taum[1]), ABS(taum[2]), ABS(taum[3]));

  taum[0] = massprod_edge2d(xm[3], xm[0]);
  taum[1] = massprod_edge2d(xm[3], xm[1]);
  taum[2] = massprod_edge2d(xm[3], xm[2]);
  taum[3] = massprod_edge2d(xm[3], xm[3]);
#pragma omp single
  printf("  %.2e %.2e %.2e %.2e\n",
	 ABS(taum[0]), ABS(taum[1]), ABS(taum[2]), ABS(taum[3]));

  norm = norm2_edge2d(b);
  tau = massprod_edge2d(xm[0], b);
  add_edge2d(-tau, xm[0], b);
  error = norm2_edge2d(b);
#pragma omp single
  printf("  First vector: %.2e (%.2e)\n",
	 error, error/norm);

  norm = norm2_edge2d(d);
  tau = massprod_edge2d(xm[0], d);
  add_edge2d(-tau, xm[0], d);
  tau = massprod_edge2d(xm[1], d);
  add_edge2d(-tau, xm[1], d);
  error = norm2_edge2d(d);
#pragma omp single
  printf("  Second vector: %.2e (%.2e)\n",
	 error, error/norm);
  
  printf("========================================\n");
  
  return 0;
}

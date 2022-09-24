
#include <math.h>
#include <stdio.h>

#include "node2d.h"
#include "edge2d.h"

static field
example0_sol(const real *x, void *data)
{
  (void) data;

  return 1.0;
}

static field
example1_sol(const real *x, void *data)
{
  (void) data;

  return sin(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
}

static field
example1_rhs(const real *x, void *data)
{
  return 8.0 * M_PI * M_PI * example1_sol(x, data);
}

static field
example2_sol(const real *x, void *data)
{
  (void) data;

  return EXP(2.0 * I * M_PI * x[0]) * EXP(4.0 * I * M_PI * x[1]);
}

static field
example2_rhs(const real *x, void *data)
{
  return 20.0 * M_PI * M_PI * example2_sol(x, data);
}

int
main(int argc, char **argv)
{
  grid2d *gr, *grf;
  node2d *x, *b, *d, *xf, *bf, *df;
  edge2d *xe, *be;
  matrix *A;
  field *xd, *bd;
  real oldnorm, norm;
  field product1, product2;
  int nx, ny;
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

      x = new_node2d(gr);
      b = new_node2d(gr);
      d = new_node2d(gr);

      xe = new_edge2d(gr);
      be = new_edge2d(gr);

      grf = new_grid2d(2*nx, 2*ny, 0.5/nx, 0.5/ny);
      grf->xfactor = gr->xfactor;
      grf->yfactor = gr->yfactor;
      
      xf = new_node2d(grf);
      bf = new_node2d(grf);
      df = new_node2d(grf);
    }

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing integration\n"
	 "----------------------------------------\n");
  l2functional_node2d(b, d, example0_sol, 0);
#pragma omp single
  {
    product1 = 0.0;
    for(j=0; j<ny; j++)
      for(i=0; i<nx; i++)
	product1 += b->v[i+j*nx];
  }
#pragma omp single
  printf("  Averaged quadrature error %.2e\n",
	 ABS(product1 - 1.0));
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing first solution\n"
	 "----------------------------------------\n");
#pragma omp single
  printf("Integrating right-hand side\n");
  l2functional_node2d(b, d, example1_rhs, 0);
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  {
    product1 = 0.0;
    for(j=0; j<ny; j++)
      for(i=0; i<nx; i++)
	product1 += b->v[i+j*nx];
  }
#pragma omp single
  printf("  Averaged quadrature error %.2e\n",
	 ABS(product1));

#pragma omp single
  printf("Interpolating solution\n");
  interpolate_node2d(x, example1_sol, 0);
  norm = norm2_node2d(x);
#pragma omp single
  printf("  Norm %.4e\n", norm);

  norm = l2norm_node2d(x, example1_sol, 0);
#pragma omp single
  printf("  L^2 error %.4e\n", norm);

#pragma omp single
  printf("Evaluating differential operator\n");
  zero_node2d(b);
  addeval_node2d(-1.0, x, b);
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Error %.4e\n", norm);
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing second solution\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Integrating right-hand side\n");
  l2functional_node2d(b, d, example2_rhs, 0);
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  {
    product1 = 0.0;
    for(j=0; j<ny; j++)
      for(i=0; i<nx; i++)
	product1 += b->v[i+j*nx];
  }
#pragma omp single
  printf("  Averaged quadrature error %.2e\n",
	 ABS(product1));

#pragma omp single
  printf("Interpolating solution\n");
  interpolate_node2d(x, example2_sol, 0);
  norm = norm2_node2d(x);
#pragma omp single
  printf("  Norm %.4e\n", norm);

  norm = l2norm_node2d(x, example2_sol, 0);
#pragma omp single
  printf("  L^2 error %.4e\n", norm);

#pragma omp single
  printf("Evaluating operator\n");
  addeval_node2d(-1.0, x, b);
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Error %.4e\n", norm);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing self-adjointness\n"
	 "----------------------------------------\n");

  random_node2d(b);
  random_node2d(d);
  zero_node2d(x);
  addeval_node2d(1.0, b, x);
  product1 = dotprod_node2d(d, x);
  oldnorm = norm2_node2d(d) * norm2_node2d(x);
  zero_node2d(x);
  addeval_node2d(1.0, d, x);
  product2 = dotprod_node2d(x, b);
  norm = norm2_node2d(x) * norm2_node2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing forward GS\n"
	 "----------------------------------------\n");
  
  zero_node2d(b);
  random_node2d(x);
  
  copy_node2d(b, d);
  addeval_node2d(-1.0, x, d);
  norm = norm2_node2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  for(i=0; i<10; i++) {
    gsforward_node2d(1.0, b, x);
    
    copy_node2d(b, d);
    addeval_node2d(-1.0, x, d);
    oldnorm = norm;
    norm = norm2_node2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing backward GS\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Testing symmetry\n");
  random_node2d(b);
  random_node2d(d);
  zero_node2d(x);
  gsforward_node2d(1.0, b, x);
  product1 = dotprod_node2d(d, x);
  oldnorm = norm2_node2d(d) * norm2_node2d(x);
  zero_node2d(x);
  gsbackward_node2d(1.0, d, x);
  product2 = dotprod_node2d(x, b);
  norm = norm2_node2d(x) * norm2_node2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);
  
  zero_node2d(b);
  random_node2d(x);
  
  copy_node2d(b, d);
  addeval_node2d(-1.0, x, d);
  norm = norm2_node2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  for(i=0; i<10; i++) {
    gsbackward_node2d(1.0, b, x);
    
    copy_node2d(b, d);
    addeval_node2d(-1.0, x, d);
    oldnorm = norm;
    norm = norm2_node2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing symmetric GS\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Testing symmetry\n");
  random_node2d(b);
  random_node2d(d);
  zero_node2d(x);
  gssymm_node2d(1.0, b, x);
  product1 = dotprod_node2d(d, x);
  oldnorm = norm2_node2d(d) * norm2_node2d(x);
  zero_node2d(x);
  gssymm_node2d(1.0, d, x);
  product2 = dotprod_node2d(x, b);
  norm = norm2_node2d(x) * norm2_node2d(b);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Error %.4e (%.4e)\n",
	 ABS(product1-product2), ABS(product1-product2)/norm);
  
  zero_node2d(b);
  random_node2d(x);
  
  copy_node2d(b, d);
  addeval_node2d(-1.0, x, d);
  norm = norm2_node2d(d);
#pragma omp single
  printf("Initial residual: %.4e\n", norm);

  for(i=0; i<10; i++) {
    gsbackward_node2d(1.0, b, x);
    
    copy_node2d(b, d);
    addeval_node2d(-1.0, x, d);
    oldnorm = norm;
    norm = norm2_node2d(d);
#pragma omp single
    printf("Step %2d residual: %.4e (%.4e)\n", i+1, norm, norm/oldnorm);
  }

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing prolongation\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vectors\n");
  random_node2d(x);
  random_node2d(d);
  norm = norm2_node2d(x);
  oldnorm = norm2_node2d(d);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);

#pragma omp single
  printf("Applying prolongation\n");
  zero_node2d(xf);
  prolongation_node2d(1.0, x, xf);
  zero_node2d(df);
  prolongation_node2d(1.0, d, df);
  norm = norm2_node2d(xf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Multiplying by the system matrix on the fine grid\n");
  zero_node2d(bf);
  addeval_node2d(1.0, xf, bf);
  norm = norm2_node2d(bf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Multiplying by the system matrix on the coarse grid\n");
  zero_node2d(b);
  addeval_node2d(1.0, x, b);
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Computing inner products\n");
  product1 = dotprod_node2d(d, b);
  product2 = dotprod_node2d(df, bf);
  oldnorm = norm2_node2d(d) * norm2_node2d(b);
  norm = norm2_node2d(df) * norm2_node2d(bf);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Difference %.4e (%.4e)\n",
	 fabs(product1-product2), fabs(product1-product2)/norm);

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing restriction\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vectors\n");
  random_node2d(x);
  random_node2d(bf);
  norm = norm2_node2d(x);
  oldnorm = norm2_node2d(bf);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);

#pragma omp single
  printf("Applying prolongation and restriction\n");
  zero_node2d(xf);
  prolongation_node2d(1.0, x, xf);
  zero_node2d(b);
  restriction_node2d(1.0, bf, b);
  norm = norm2_node2d(xf);
  oldnorm = norm2_node2d(b);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);
  
#pragma omp single
  printf("Computing inner products\n");
  product1 = dotprod_node2d(x, b);
  product2 = dotprod_node2d(xf, bf);
#pragma omp single
  printf("  Difference %.4e (%.4e)\n",
	 fabs(product1-product2), fabs(product1-product2)/fabs(product1));

#pragma omp single
  printf("----------------------------------------\n"
	 "Testing Galerkin property\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vector\n");
  random_node2d(x);
  norm = norm2_node2d(x);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Applying prolongation\n");
  zero_node2d(xf);
  prolongation_node2d(1.0, x, xf);
  norm = norm2_node2d(xf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Multiplying by the system matrix on the fine grid\n");
  zero_node2d(bf);
  addeval_node2d(1.0, xf, bf);
  norm = norm2_node2d(bf);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Applying restriction\n");
  zero_node2d(b);
  restriction_node2d(1.0, bf, b);
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Norm %.4e\n", norm);

#pragma omp single
  printf("Subtracting coarse matrix-vector product\n");
  addeval_node2d(-1.0, x, b);
  oldnorm = norm;
  norm = norm2_node2d(b);
#pragma omp single
  printf("  Norm %.4e (%.4e)\n", norm, norm/oldnorm);
  
#pragma omp single
  printf("----------------------------------------\n"
	 "Testing gradient lifting\n"
	 "----------------------------------------\n");

#pragma omp single
  printf("Creating random vectors\n");
  random_node2d(x);
  random_edge2d(be);
  norm = norm2_node2d(x);
  oldnorm = norm2_edge2d(be);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);

#pragma omp single
  printf("Applying gradient and adjoint gradient\n");
  zero_edge2d(xe);
  gradient_node2d(1.0, x, xe);
  zero_node2d(b);
  adjgradient_node2d(1.0, be, b);
  norm = norm2_edge2d(xe);
  oldnorm = norm2_node2d(b);
#pragma omp single
  printf("  Norms %.4e and %.4e\n", norm, oldnorm);
  
#pragma omp single
  printf("Computing inner products\n");
  product1 = dotprod_node2d(x, b);
  oldnorm = norm2_node2d(x) * norm2_node2d(b);
  product2 = dotprod_edge2d(xe, be);
  norm = norm2_edge2d(xe) * norm2_edge2d(be);
  if(oldnorm > norm)
    norm = oldnorm;
#pragma omp single
  printf("  Difference %.4e (%.4e)\n",
	 fabs(product1-product2), fabs(product1-product2)/norm);
  }

  if(nx * ny <= 65536) {
#pragma omp single
    printf("----------------------------------------\n"
	   "Testing gradient elimination\n"
	   "----------------------------------------\n");
    
#pragma omp single
    printf("Creating random vector\n");
    random_edge2d(xe);
    norm = norm2_edge2d(xe);
#pragma omp single
    printf("  Norm %.4e\n", norm);
    
#pragma omp single
    printf("Applying mass matrix\n");
    zero_edge2d(be);
    addeval_edge2d(0.0, 1.0, xe, be);
    norm = norm2_edge2d(be);
#pragma omp single
    printf("  Norm %.4e\n", norm);
    
#pragma omp single
    printf("Applying adjoint gradient\n");
    zero_node2d(b);
    adjgradient_node2d(1.0, be, b);
    oldnorm = norm2_node2d(b);
#pragma omp single
    printf("  Norm %.4e\n", oldnorm);
    
#pragma omp single
    printf("Gauss-Seidel solver\n");
    zero_node2d(x);
    for(i=0; i<5000; i++)
      gsforward_node2d(1.0, b, x);
    copy_node2d(b, d);
    addeval_node2d(-1.0, x, d);
    norm = norm2_node2d(d);
#pragma omp single
    printf("  Defect norm %.4e\n", norm);
    
#pragma omp single
    printf("Adding correction\n");
    gradient_node2d(-1.0, x, xe);
    norm = norm2_edge2d(xe);
#pragma omp single
    printf("  Norm %.4e\n", norm);
    
#pragma omp single
    printf("Applying mass matrix\n");
    zero_edge2d(be);
    addeval_edge2d(0.0, 1.0, xe, be);
    norm = norm2_edge2d(be);
#pragma omp single
    printf("  Norm %.4e\n", norm);
    
#pragma omp single
    printf("Applying adjoint gradient\n");
    zero_node2d(b);
    adjgradient_node2d(1.0, be, b);
    norm = norm2_node2d(b);
#pragma omp single
    printf("  Norm %.4e (%.4e)\n", norm, norm/oldnorm);
  }
    
  if(nx*ny <= 8192) {
#pragma omp single
    printf("----------------------------------------\n"
	   "Testing dense matrix conversion\n"
	   "----------------------------------------\n");
    A = 0;
    xd = bd = 0;
#pragma omp single
    {
      A = new_matrix(nx*ny, nx*ny);
      xd = (field *) malloc(sizeof(field) * nx*ny);
      bd = (field *) malloc(sizeof(field) * nx*ny);
      densematrix_node2d(false, gr, A);
    }

#pragma omp single
    printf("Safe method\n");
    
    random_node2d(x);
    
#pragma omp single
    {
      densefrom_node2d(x, xd);
      for(i=0; i<nx*ny; i++)
	bd[i] = 0.0;
      for(j=0; j<nx*ny; j++)
	for(i=0; i<nx*ny; i++)
	  bd[i] += A->a[i+j*A->ld] * xd[j];
      denseto_node2d(bd, b);
    }
    oldnorm = norm2_node2d(b);
    addeval_node2d(-1.0, x, b);
    norm = norm2_node2d(b);

#pragma omp single
    {
      printf("  Error %.2e (%.2e)\n",
	     norm, norm/oldnorm);
    }

#pragma omp single
    printf("Simple method\n");
    
    random_node2d(x);
    
#pragma omp single
    {
      for(i=0; i<nx*ny; i++)
	b->v[i] = 0.0;
      for(j=0; j<nx*ny; j++)
	for(i=0; i<nx*ny; i++)
	  b->v[i] += A->a[i+j*A->ld] * x->v[j];
    }
    oldnorm = norm2_node2d(b);
    addeval_node2d(-1.0, x, b);
    norm = norm2_node2d(b);

#pragma omp single
    {
      printf("  Error %.2e (%.2e)\n",
	     norm, norm/oldnorm);
      free(bd);
      free(xd);
      free(A);
    }
}

  printf("========================================\n");
  
  return 0;
}

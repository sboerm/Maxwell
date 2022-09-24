
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "edge2d.h"
#include "node2d.h"
#include "basic.h"
#include "linalg.h"

int
main(int argc, char **argv)
{
  grid2d **gr;
  edge2d **e, **x, **b, **d;
  node2d **xg, **bg, **dg;
  matrix *A, *M;
  real **lambda;
  real rayleigh;
  field product1, product2;
  real oldnorm, norm, dnorm;
  real t_start, t_stop, t_run;
  char filename[20];
  size_t sz;
  int pinvit_steps = 5;
  int prec_steps = 2;
  int gradient_steps = 3;
  int n, nl;
  int L;
  int ev;
  int i, j, l;

  L = 7;

  if(argc > 1)
    sscanf(argv[1], "%d", &L);

  n = 8;

  if(argc > 2)
    sscanf(argv[2], "%d", &n);

  printf("========================================\n"
	 "Eigenvalue tests, %d levels\n"
	 "========================================\n", L+1);
  
  gr = (grid2d **) malloc(sizeof(grid2d *) * (L+1));
  e = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  x = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  b = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  d = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  xg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  bg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  dg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  lambda = (real **) malloc(sizeof(real *) * (L+1));
  
  gr[L] = new_grid2d(n << L, n << L, 1.0 / (n << L), 1.0 / (n << L));
  nl = n << L;
  for(j=0; j<nl; j++)
    for(i=0; i<nl; i++)
      gr[L]->eps[i+j*nl] = ((i-nl/2)*(i-nl/2) + (j-nl/2)*(j-nl/2) <= nl*nl/9 ?
			   100.0 : 1.0);

  if(L <= 5) {
    snprintf(filename, 20, "grid%02d.png", L);
    draw2d_grid2d(gr[L], filename);
  }
  
  for(l=L; l-->0; ) {
    gr[l] = coarsen_grid2d(gr[l+1]);

    if(l <= 5) {
      snprintf(filename, 20, "grid%02d.png", l);
      draw2d_grid2d(gr[l], filename);
    }
  }
  
  for(l=0; l<=L; l++) {
    e[l] = new_edge2d(gr[l]);
    x[l] = new_edge2d(gr[l]);
    b[l] = new_edge2d(gr[l]);
    d[l] = new_edge2d(gr[l]);
    xg[l] = new_node2d(gr[l]);
    bg[l] = new_node2d(gr[l]);
    dg[l] = new_node2d(gr[l]);
  }
  
  lambda[0] = (real *) malloc(sizeof(real) * 2 * n * (n-1));
  for(l=1; l<=L; l++)
    lambda[l] = 0;
  
  for(l=0; l<=L; l++) {
    sz = (sizeof(field) * gr[l]->nx * (gr[l]->ny+1)
	  + sizeof(field) * (gr[l]->nx+1) * gr[l]->ny
	  + sizeof(edge2d));
    printf("Level %2d: %d x %d grid, %zd dofs, %.1f MB per gridfunc\n",
	   l, gr[l]->nx, gr[l]->ny,
	   (size_t) gr[l]->nx * (gr[l]->ny-1)
	   + (size_t) gr[l]->ny * (gr[l]->nx-1),
	   sz / 1048576.0);
  }
  
  printf("Setting up stiffness and mass matrix\n");
  t_start = omp_get_wtime();
  A = new_matrix(2*n*n, 2*n*n);
  M = new_matrix(2*n*n, 2*n*n);
  densematrix_edge2d(1.0, 0.0, gr[0], A);
  densematrix_edge2d(0.0, 1.0, gr[0], M);
  t_stop = omp_get_wtime();
  t_run = t_stop - t_start;
  printf("  %.2f seconds\n", t_run);
  
  printf("Solving coarse eigenvalue problem\n");
  t_start = omp_get_wtime();
  sygv_matrix(A, M, lambda[0]);
  t_stop = omp_get_wtime();
  t_run = t_stop - t_start;
  printf("  %.2f seconds\n", t_run);
  for(i=0; i<A->rows && fabs(lambda[0][i]) < 1e-8; i++)
    ;
  ev = i;
  printf("%d eigenvalues, %d zeros, then %f %f %f %f\n",
	 A->rows, ev, lambda[0][ev], lambda[0][ev+1],
	 lambda[0][ev+2], lambda[0][ev+3]);
  
  denseto_edge2d(A->a+ev*A->ld, e[0]);

#pragma omp parallel private(i,j,l,t_start,t_stop,t_run,product1,product2,rayleigh,norm,oldnorm)
  {
    for(l=1; l<=L; l++) {
      t_start = omp_get_wtime();
      
      /* Transfer solution from coarser mesh */
      zero_edge2d(e[l]);
      prolongation_edge2d(1.0, e[l-1], e[l]);

      /* Compute A e */
      zero_edge2d(b[l]);
      addeval_edge2d(1.0, 1.0, e[l], b[l]);

      /* Compute M e */
      zero_edge2d(d[l]);
      addeval_edge2d(0.0, 1.0, e[l], d[l]);

      /* Compute the Rayleigh quotient */
      product1 = dotprod_edge2d(e[l], b[l]);
      product2 = dotprod_edge2d(e[l], d[l]);
      rayleigh = product1 / product2;

#pragma omp single
      printf("----------------------------------------\n"
	     "Level %d, initial Rayleigh quotient %f\n",
	     l, rayleigh-1.0);

      /* Compute the defect A e - lambda M e */
      add_edge2d(-rayleigh, d[l], b[l]);

      norm = norm2_edge2d(b[l]);

      /* Estimate the dual norm */
      zero_edge2d(x[l]);
      vcycle_edge2d(l, 2, 0, b, x, d);
      dnorm = sqrt(fabs(dotprod_edge2d(x[l], b[l]))) / norm;
      
#pragma omp single
      printf("Initial defect %.4e (%.4e)\n", norm, dnorm);

      for(i=0; i<pinvit_steps; i++) {
	/* Apply the multigrid preconditioner */
	zero_edge2d(x[l]);
	for(j=0; j<prec_steps; j++)
	  vcycle_edge2d(l, 2, 0, b, x, d);
	
	/* Add update to eigenvector approximation */
	add_edge2d(-1.0, x[l], e[l]);
	  
	/* Eliminate null-space components */
	center_edge2d(e[l]);
	
	zero_edge2d(d[l]);
	addeval_edge2d(0.0, 1.0, e[l], d[l]);
	zero_node2d(bg[l]);
	adjgradient_node2d(1.0, d[l], bg[l]);
	zero_node2d(xg[l]);
	for(j=0; j<gradient_steps; j++)
	  vcycle_node2d(l, 2, 0, bg, xg, dg);
	gradient_node2d(-1.0, xg[l], e[l]);
	
	/* Make it a unit vector */
	zero_edge2d(d[l]);
	addeval_edge2d(0.0, 1.0, e[l], d[l]);
	product1 = dotprod_edge2d(e[l], d[l]);
	scale_edge2d(1.0/sqrt(product1), e[l]);
	
	/* Compute A e */
	zero_edge2d(b[l]);
	addeval_edge2d(1.0, 1.0, e[l], b[l]);

	/* Evaluate the Rayleigh quotient */
	rayleigh = dotprod_edge2d(e[l], b[l]);

	/* Compute M e */
	addeval_edge2d(0.0, -rayleigh, e[l], b[l]);

	oldnorm = norm;
	norm = norm2_edge2d(b[l]);

#pragma omp single
	printf("Step %2d defect: %.4e (%.3f), Rayleigh %f\n",
	       i+1, norm, norm/oldnorm, rayleigh-1.0);
      }

      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for level %d\n", t_run, l);
    }
  }
  
  return 0;
}

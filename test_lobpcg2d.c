
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "edge2d.h"
#include "node2d.h"
#include "basic.h"
#include "linalg.h"

int
main(int argc, char **argv)
{
  grid2d **gr;
  edge2d ***e, ***up, ***eold, ***enew;
  edge2d **x, **b, **d;
  edge2d *aux, **basis;
  node2d **xg, **bg, **dg;
  matrix *A, *M;
  matrix *Ar2, *Ar3;
  real **lambda;
  field product1, product2;
  field *tau;
  real *oldnorm, *norm, *dnorm;
  real t_start, t_stop, t_run;
  char filename[20];
  size_t sz;
  int pinvit_steps = 5;
  int prec_steps = 2;
  int gradient_steps = 4;
  int n, nl;
  int L, eigs;
  int ev;
  int i, j, k, l, info;

  L = 7;

  if(argc > 1)
    sscanf(argv[1], "%d", &L);

  n = 8;

  if(argc > 2)
    sscanf(argv[2], "%d", &n);

  eigs = 4;

  if(argc > 3)
    sscanf(argv[3], "%d", &eigs);
  
  printf("========================================\n"
	 "Eigenvalue tests, %d levels\n"
	 "========================================\n", L+1);
  
  gr = (grid2d **) malloc(sizeof(grid2d *) * (L+1));
  e = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  up = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  eold = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  enew = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  x = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  b = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  d = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  xg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  bg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  dg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  lambda = (real **) malloc(sizeof(real *) * (L+1));
  basis = (edge2d **) malloc(sizeof(edge2d *) * 3 * eigs);
  tau = (field *) malloc(sizeof(field) * 3 * eigs);
  
  gr[L] = new_grid2d(n << L, n << L, 1.0 / (n << L), 1.0 / (n << L));
  nl = n << L;
  for(j=0; j<nl; j++)
    for(i=0; i<nl; i++)
      gr[L]->eps[i+j*nl] = ((i-nl/2)*(i-nl/2) + (j-nl/2)*(j-nl/2) <= nl*nl/9 ?
			   100.0 : 1.0);

  gr[L]->xfactor = I;
  
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
    e[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      e[l][i] = new_edge2d(gr[l]);
    up[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      up[l][i] = new_edge2d(gr[l]);
    eold[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      eold[l][i] = new_edge2d(gr[l]);
    enew[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      enew[l][i] = new_edge2d(gr[l]);
    x[l] = new_edge2d(gr[l]);
    b[l] = new_edge2d(gr[l]);
    d[l] = new_edge2d(gr[l]);
    xg[l] = new_node2d(gr[l]);
    bg[l] = new_node2d(gr[l]);
    dg[l] = new_node2d(gr[l]);
  }
  
  lambda[0] = (real *) malloc(sizeof(real) * 2 * n * (n-1));
  for(l=1; l<=L; l++)
    lambda[l] = (real *) malloc(sizeof(real) * 3 * eigs);

  norm = (real *) malloc(sizeof(real) * eigs);
  oldnorm = (real *) malloc(sizeof(real) * eigs);
  dnorm = (real *) malloc(sizeof(real) * eigs);
  
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

  Ar2 = new_matrix(2*eigs, 2*eigs);
  Ar3 = new_matrix(3*eigs, 3*eigs);
  
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

  for(i=0; i<eigs; i++)
    denseto_edge2d(A->a+(ev+i)*A->ld, e[0][i]);

#pragma omp parallel private(i,j,k,l,info,t_start,t_stop,t_run,product1,product2)
  {
    for(l=1; l<=L; l++) {
      t_start = omp_get_wtime();
      
      for(i=0; i<eigs; i++) {
	/* Transfer solution from coarser mesh */
	zero_edge2d(e[l][i]);
	prolongation_edge2d(1.0, e[l-1][i], e[l][i]);
	
	/* Compute A e */
	zero_edge2d(b[l]);
	addeval_edge2d(1.0, 1.0, e[l][i], b[l]);
	
	/* Compute M e */
	zero_edge2d(d[l]);
	addeval_edge2d(0.0, 1.0, e[l][i], d[l]);
	
	/* Compute the Rayleigh quotient */
	product1 = dotprod_edge2d(e[l][i], b[l]);
	product2 = dotprod_edge2d(e[l][i], d[l]);
	lambda[l][i] = product1 / product2;

	/* Compute the defect A e - lambda M e */
	add_edge2d(-lambda[l][i], d[l], b[l]);
	
	norm[i] = norm2_edge2d(b[l]);
	
	/* Estimate the dual norm */
	zero_edge2d(x[l]);
	vcycle_edge2d(l, 2, 0, b, x, d);
	dnorm[i] = sqrt(fabs(dotprod_edge2d(x[l], b[l]))) / norm[i];
      }

#pragma omp single
      {
	printf("----------------------------------------\n"
	       "Level %2d:", l);
	for(i=0; i<eigs; i++)
	  printf(" %f", lambda[l][i] - 1.0);
	printf("\n");

	printf(" Defects:");
	for(i=0; i<eigs; i++)
	  printf(" %.2e", norm[i]);
	printf("\n");

	printf("   Duals:");
	for(i=0; i<eigs; i++)
	  printf(" %.2e", dnorm[i]);
	printf("\n");
      }

      for(k=0; k<pinvit_steps; k++) {
	for(i=0; i<eigs; i++) {
	  /* Compute the defect */
	  zero_edge2d(b[l]);
	  addeval_edge2d(1.0, 1.0-lambda[l][i], e[l][i], b[l]);

	  /* Apply the multigrid preconditioner */
	  zero_edge2d(x[l]);
	  for(j=0; j<prec_steps; j++)
	    vcycle_edge2d(l, 2, 0, b, x, d);
	  
	  /* Eliminate null-space components */
	  if(gr[l]->xfactor == 1.0 && gr[l]->yfactor == 1.0)
	    center_edge2d(x[l]);
	  
	  zero_edge2d(d[l]);
	  addeval_edge2d(0.0, 1.0, x[l], d[l]);
	  zero_node2d(bg[l]);
	  adjgradient_node2d(1.0, d[l], bg[l]);
	  zero_node2d(xg[l]);
	  for(j=0; j<gradient_steps; j++)
	    vcycle_node2d(l, 2, 0, bg, xg, dg);
	  gradient_node2d(-1.0, xg[l], x[l]);
	
	  /* Copy to update array */
	  copy_edge2d(x[l], up[l][i]);
	}

#pragma omp barrier

	if(k == 0) {
	  /* Set up basis */
	  for(i=0; i<eigs; i++) {
#ifdef TRUE_LOBPCG
	    basis[i] = e[l][i];
	    basis[i+eigs] = up[l][i];
#else
	    basis[i] = up[l][i];
	    basis[i+eigs] = e[l][i];
#endif
	  }
	  orthonormalize_edge2d(2*eigs, basis, tau, d[l]);
	
	  /* Set up Ritz system */
	  for(j=0; j<2*eigs; j++)
	    for(i=j; i<2*eigs; i++)
	      Ar2->a[i+Ar2->ld*j] = energyprod_edge2d(1.0, 1.0, basis[i], basis[j]);

	  /* Solve Ritz system */
#pragma omp single
	  {
	    info = syev_matrix(Ar2, lambda[l]);
	    assert(info == 0);
	  }
	  
	  /* Construct eigenvector approximations */
	  for(j=0; j<eigs; j++) {
	    zero_edge2d(enew[l][j]);
	    for(i=0; i<2*eigs; i++)
	      add_edge2d(Ar2->a[i+Ar2->ld*j], basis[i], enew[l][j]);
	  }
	}
	else {
	  /* Set up basis */
	  for(i=0; i<eigs; i++) {
#ifdef TRUE_LOBPCG
	    basis[i] = eold[l][i];
	    basis[i+eigs] = e[l][i];
	    basis[i+2*eigs] = up[l][i];
#else
	    basis[i] = up[l][i];
	    basis[i+eigs] = e[l][i];
	    basis[i+2*eigs] = eold[l][i];
#endif
	  }
	  orthonormalize_edge2d(3*eigs, basis, tau, d[l]);
	  
	  /* Set up Ritz system */
	  for(j=0; j<3*eigs; j++)
	    for(i=j; i<3*eigs; i++)
	      Ar3->a[i+Ar3->ld*j] = energyprod_edge2d(1.0, 1.0, basis[i], basis[j]);

	  /* Solve Ritz system */
#pragma omp single
	  {
	    info = syev_matrix(Ar3, lambda[l]);
	    assert(info == 0);
	  }

	  /* Construct eigenvector approximations */
	  for(j=0; j<eigs; j++) {
	    zero_edge2d(enew[l][j]);
	    for(i=0; i<3*eigs; i++)
	      add_edge2d(Ar3->a[i+Ar3->ld*j], basis[i], enew[l][j]);
	  }
	}
	
	/* Switch old and new approximations */
#pragma omp single
	for(i=0; i<eigs; i++) {
	  aux = eold[l][i];
	  eold[l][i] = e[l][i];
	  e[l][i] = enew[l][i];
	  enew[l][i] = aux;
	}

	/* Eliminate null-space components */
	for(i=0; i<eigs; i++) {
	  if(gr[l]->xfactor == 1.0 && gr[l]->yfactor == 1.0)
	    center_edge2d(e[l][i]);
	  
	  zero_edge2d(d[l]);
	  addeval_edge2d(0.0, 1.0, e[l][i], d[l]);
	  zero_node2d(bg[l]);
	  adjgradient_node2d(1.0, d[l], bg[l]);
	  zero_node2d(xg[l]);
	  for(j=0; j<gradient_steps; j++)
	    vcycle_node2d(l, 2, 0, bg, xg, dg);
	  gradient_node2d(-1.0, xg[l], e[l][i]);
	}
	
	/* Compute defects */
	for(i=0; i<eigs; i++) {
	  zero_edge2d(d[l]);
	  addeval_edge2d(1.0, 1.0-lambda[l][i], e[l][i], d[l]);
	  oldnorm[i] = norm[i];
	  norm[i] = norm2_edge2d(d[l]);
	}

#pragma omp single
	{
	  printf(" Step %2d:", k+1);
	  for(i=0; i<eigs; i++)
	    printf(" %f", lambda[l][i] - 1.0);
	  printf("\n");

	  printf(" Defects:");
	  for(i=0; i<eigs; i++)
	    printf(" %.2e", norm[i]);
	  printf("\n");
	  
	  printf("   Rates:");
	  for(i=0; i<eigs; i++)
	    printf(" %.3f", norm[i]/oldnorm[i]);
	  printf("\n");

	  fflush(stdout);
	}
      }

      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for level %d\n", t_run, l);
    }
  }
  
  return 0;
}

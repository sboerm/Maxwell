
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "edge2d.h"
#include "basic.h"

int
main(int argc, char **argv)
{
  grid2d **gr;
  edge2d *cgx, *cgb, *cgp, *cga;
  edge2d **x, **b, **d;
  edge2d *xtest, *ytest;
  real oldnorm, norm, resnorm, product1, product2;
  real t_start, t_stop, t_run;
  size_t sz;
  int iterations = 30;
  int n, nl;
  int L;
  int i, j, l;

  L = 7;

  if(argc > 1)
    sscanf(argv[1], "%d", &L);

  n = 2;

  if(argc > 2)
    sscanf(argv[2], "%d", &n);

  printf("========================================\n"
	 "Multigrid tests, %d levels\n"
	 "========================================\n", L+1);
  
  gr = (grid2d **) malloc(sizeof(grid2d *) * (L+1));
  x = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  b = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  d = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  
  gr[L] = new_grid2d(n << L, n << L, 1.0 / (n << L), 1.0 / (n << L));
  nl = n << L;
  for(j=0; j<nl; j++)
    for(i=0; i<nl; i++)
      gr[L]->eps[i+j*nl] = ((i-nl/2)*(i-nl/2) + (j-nl/2)*(j-nl/2) <= nl*nl/9 ?
			    100.0 : 1.0);
  
  for(l=L; l-->0; )
    gr[l] = coarsen_grid2d(gr[l+1]);
  
  for(l=0; l<=L; l++) {
    x[l] = new_edge2d(gr[l]);
    b[l] = new_edge2d(gr[l]);
    d[l] = new_edge2d(gr[l]);
  }
  
  cgx = new_edge2d(gr[L]);
  cgb = new_edge2d(gr[L]);
  cgp = new_edge2d(gr[L]);
  cga = new_edge2d(gr[L]);
  
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

#pragma omp parallel private(i,l,norm,oldnorm)
  {
#pragma omp single
    printf("----------------------------------------\n"
	   "Testing Galerkin property\n"
	   "----------------------------------------\n");
    for(l=1; l<=L; l++) {
      random_edge2d(x[l-1]);

      zero_edge2d(x[l]);
      prolongation_edge2d(1.0, x[l-1], x[l]);

      zero_edge2d(b[l]);
      addeval_edge2d(1.0, 1.0, x[l], b[l]);

      zero_edge2d(b[l-1]);
      restriction_edge2d(1.0, b[l], b[l-1]);

      oldnorm = norm2_edge2d(b[l-1]);

      addeval_edge2d(-1.0, -1.0, x[l-1], b[l-1]);

      norm = norm2_edge2d(b[l-1]);
#pragma omp single
      printf("Level %2d: Error %.4e (%.4e)\n", l, norm, norm/oldnorm);
    }

#pragma omp single
    printf("----------------------------------------\n"
	   "Testing symmetry of the GS preconditioner\n"
	   "----------------------------------------\n");
    for(l=0; l<=L; l++) {
#pragma omp single
      {
	xtest = new_edge2d(gr[l]);
	ytest = new_edge2d(gr[l]);
      }
      
      random_edge2d(xtest);
      random_edge2d(ytest);

      copy_edge2d(xtest, b[l]);
      zero_edge2d(x[l]);
      gsforward_edge2d(1.0, 1.0, b[l], x[l]);
      product1 = dotprod_edge2d(x[l], ytest);
      oldnorm = norm2_edge2d(x[l]) * norm2_edge2d(ytest);

      copy_edge2d(ytest, b[l]);
      zero_edge2d(x[l]);
      gsbackward_edge2d(1.0, 1.0, b[l], x[l]);
      product2 = dotprod_edge2d(xtest, x[l]);
      norm = norm2_edge2d(xtest) * norm2_edge2d(x[l]);

      if(oldnorm > norm)
	norm = oldnorm;

#pragma omp single
      {
	del_edge2d(ytest);
	del_edge2d(xtest);
      }
      
#pragma omp single
      printf("Level %2d: Error %.4e (%.4e)\n",
	     l, fabs(product1-product2),
	     fabs(product1-product2) / norm);
    }
    
#pragma omp single
    printf("----------------------------------------\n"
	   "Testing symmetry of the multigrid preconditioner\n"
	   "----------------------------------------\n");
    l = L;
    for(l=0; l<=L; l++) {
#pragma omp single
      {
	xtest = new_edge2d(gr[l]);
	ytest = new_edge2d(gr[l]);
      }
      
      random_edge2d(xtest);
      random_edge2d(ytest);

      copy_edge2d(xtest, b[l]);
      zero_edge2d(x[l]);
      vcycle_edge2d(l, 2, 0, b, x, d);
      product1 = dotprod_edge2d(x[l], ytest);
      oldnorm = norm2_edge2d(x[l]) * norm2_edge2d(ytest);

      copy_edge2d(ytest, b[l]);
      zero_edge2d(x[l]);
      vcycle_edge2d(l, 2, 0, b, x, d);
      product2 = dotprod_edge2d(xtest, x[l]);
      norm = norm2_edge2d(xtest) * norm2_edge2d(x[l]);

      if(oldnorm > norm)
	norm = oldnorm;

#pragma omp single
      {
	del_edge2d(ytest);
	del_edge2d(xtest);
      }

#pragma omp single
      printf("Level %2d: Error %.4e (%.4e)\n",
	     l, fabs(product1-product2),
	     fabs(product1-product2) / norm);
    }
    
#pragma omp single
    printf("----------------------------------------\n"
	   "Testing V-cycle multigrid\n"
	   "----------------------------------------\n");

#pragma omp single
    printf("Creating random start vector\n");
    zero_edge2d(b[L]);
    random_edge2d(x[L]);
    
    copy_edge2d(b[L], d[L]);
    addeval_edge2d(-1.0, -1.0, x[L], d[L]);
    norm = norm2_edge2d(d[L]);
#pragma omp single
    printf("Initial residual: %.4e\n", norm);

    t_start = omp_get_wtime();
    for(i=0; i<iterations; i++) {
      vcycle_edge2d(L, 2, 0, b, x, d);
      
      copy_edge2d(b[L], d[L]);
      addeval_edge2d(-1.0, -1.0, x[L], d[L]);
      oldnorm = norm;
      norm = norm2_edge2d(d[L]);

#pragma omp single
      printf("Step %2d residual: %.4e (%.3f)\n", i+1, norm, norm/oldnorm);
    }
    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("%.2f seconds, %.2f seconds per iteration\n",
	   t_run, t_run / 20);

    norm = norm2_edge2d(x[L]);
#pragma omp single
    printf("Error norm %.4e\n", norm);

#pragma omp single
    printf("----------------------------------------\n"
	   "Testing CG with multigrid preconditioner\n"
	   "----------------------------------------\n");
    
#pragma omp single
    printf("Creating random start vector\n");
    zero_edge2d(cgb);
    random_edge2d(cgx);
    
    copy_edge2d(cgb, d[L]);
    addeval_edge2d(-1.0, -1.0, cgx, d[L]);
    norm = norm2_edge2d(d[L]);
#pragma omp single
    printf("Initial residual: %.4e\n", norm);

    mgcginit_edge2d(L, 2, cgb, cgx, cgp, cga, b, x, d);

    t_start = omp_get_wtime();
    for(i=0; i<iterations; i++) {
      mgcgstep_edge2d(L, 2, cgb, cgx, cgp, cga, b, x, d);

      resnorm = norm2_edge2d(b[L]);

      copy_edge2d(cgb, d[L]);
      addeval_edge2d(-1.0, -1.0, cgx, d[L]);
      oldnorm = norm;
      norm = norm2_edge2d(d[L]);

#pragma omp single
      printf("Step %2d residual: %.4e (%.3f), updated %.4e\n", i+1, norm, norm/oldnorm, resnorm);
    }
    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("%.2f seconds, %.2f seconds per iteration\n",
	   t_run, t_run / 20);

    norm = norm2_edge2d(cgx);
#pragma omp single
    printf("Error norm %.4e\n", norm);
  }

  return 0;
}

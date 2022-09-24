
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "edge2d.h"
#include "node2d.h"
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
  grid2d **gr;
  edge2d *cgx, *cgb, *cgp, *cga, **x, **b, **d;
  node2d **xg, **bg;
  real l2norm, oldnorm, norm;
  real t_start, t_stop, t_run;
  size_t sz;
  int n;
  int L;
  int i, l;

  L = 7;

  if(argc > 1)
    sscanf(argv[1], "%d", &L);

  n = 2;

  if(argc > 2)
    sscanf(argv[2], "%d", &n);

#pragma omp parallel private(i,l,norm,l2norm,oldnorm)
  {
#pragma omp single
    {
      printf("========================================\n"
	     "Multigrid tests, %d levels\n"
	     "========================================\n", L+1);
      
      gr = (grid2d **) malloc(sizeof(grid2d *) * (L+1));
      x = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
      b = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
      d = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
      xg = (node2d **) malloc(sizeof(node2d *) * (L+1));
      bg = (node2d **) malloc(sizeof(node2d *) * (L+1));

      gr[0] = new_grid2d(n, n, 1.0/n, 1.0/n);
      for(l=1; l<=L; l++)
	gr[l] = new_grid2d(2 * gr[l-1]->nx, 2 * gr[l-1]->ny,
			   0.5 * gr[l-1]->hx, 0.5 * gr[l-1]->hy);
      
      for(l=0; l<=L; l++) {
	x[l] = new_edge2d(gr[l]);
	b[l] = new_edge2d(gr[l]);
	d[l] = new_edge2d(gr[l]);
	xg[l] = new_node2d(gr[l]);
	bg[l] = new_node2d(gr[l]);
      }

      cgx = new_edge2d(gr[L]);
      cgb = new_edge2d(gr[L]);
      cgp = new_edge2d(gr[L]);
      cga = new_edge2d(gr[L]);
      
      for(l=0; l<=L; l++) {
	sz = getsize_edge2d(x[l]);
	printf("Level %2d: %d x %d grid, %zd dofs, %.1f MB per gridfunc\n",
	       l, gr[l]->nx, gr[l]->ny,
	       getdimension_edge2d(x[l]),
	       sz / 1048576.0);
      }
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
    for(i=0; i<10; i++) {
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
	   t_run, t_run / 10);

#pragma omp single
    printf("----------------------------------------\n"
	   "Testing Hiptmair V-cycle multigrid\n"
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
    for(i=0; i<10; i++) {
      hcycle_edge2d(L, 2, 0, b, x, d, bg, xg);
      
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
	   t_run, t_run / 10);

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
    for(i=0; i<10; i++) {
      mgcgstep_edge2d(L, 2, cgb, cgx, cgp, cga, b, x, d);
      
      copy_edge2d(cgb, d[L]);
      addeval_edge2d(-1.0, -1.0, cgx, d[L]);
      oldnorm = norm;
      norm = norm2_edge2d(d[L]);

#pragma omp single
      printf("Step %2d residual: %.4e (%.3f)\n", i+1, norm, norm/oldnorm);
    }
    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("%.2f seconds, %.2f seconds per iteration\n",
	   t_run, t_run / 10);

#pragma omp single
    printf("----------------------------------------\n"
	   "Testing nested iteration, first example\n"
	   "----------------------------------------\n");
    
    t_start = omp_get_wtime();
    l2functional_edge2d(b[L], x[L], d[L], example1_rhs, 0);
    for(l=L; l>0; l--) {
      zero_edge2d(b[l-1]);
      restriction_edge2d(1.0, b[l], b[l-1]);
    }
    zero_edge2d(x[0]);
    gsforward_edge2d(1.0, 1.0, b[0], x[0]);
    gsbackward_edge2d(1.0, 1.0, b[0], x[0]);
    
    copy_edge2d(b[0], d[0]);
    addeval_edge2d(-1.0, -1.0, x[0], d[0]);
    norm = norm2_edge2d(d[0]);
    l2norm = l2norm_edge2d(x[0], example1_sol, 0);
#pragma omp single
    printf("Level  0: residual %.2e, L^2 error %.2e\n",
	   norm, l2norm);
    
    for(i=1; i<=L; i++) {
      zero_edge2d(x[i]);
      prolongation_edge2d(1.0, x[i-1], x[i]);
      
      vcycle_edge2d(i, 2, 0, b, x, d);
      
      copy_edge2d(b[i], d[i]);
      addeval_edge2d(-1.0, -1.0, x[i], d[i]);

      oldnorm = norm;
      norm = norm2_edge2d(d[i]);
      oldnorm = l2norm;
      l2norm = l2norm_edge2d(x[i], example1_sol, 0);
#pragma omp single
      printf("Level %2d: residual %.2e, L^2 error %.2e (%.2f)\n",
	     i, norm, l2norm, l2norm/oldnorm);
    }    

    l2norm = l2norm_edge2d(x[L], example1_sol, 0);
    interpolate_edge2d(d[L], example1_sol, 0);
    norm = l2norm_edge2d(d[L], example1_sol, 0);
#pragma omp single
    printf("Final error %.4e, interpolation error %.4e\n",
	   l2norm, norm);
    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("%.2f seconds, %.2f seconds per iteration\n",
	   t_run, t_run / 10);

#pragma omp single
    printf("----------------------------------------\n"
	   "Testing nested iteration, second example\n"
	   "----------------------------------------\n");
    
    t_start = omp_get_wtime();
    l2functional_edge2d(b[L], x[L], d[L], example2_rhs, 0);
    for(l=L; l>0; l--) {
      zero_edge2d(b[l-1]);
      restriction_edge2d(1.0, b[l], b[l-1]);
    }
    zero_edge2d(x[0]);
    gsforward_edge2d(1.0, 1.0, b[0], x[0]);
    gsbackward_edge2d(1.0, 1.0, b[0], x[0]);
    
    copy_edge2d(b[0], d[0]);
    addeval_edge2d(-1.0, -1.0, x[0], d[0]);
    norm = norm2_edge2d(d[0]);
    l2norm = l2norm_edge2d(x[0], example2_sol, 0);
#pragma omp single
    printf("Level  0: residual %.2e, L^2 error %.2e\n",
	   norm, l2norm);
    
    for(i=1; i<=L; i++) {
      zero_edge2d(x[i]);
      prolongation_edge2d(1.0, x[i-1], x[i]);
      
      vcycle_edge2d(i, 2, 0, b, x, d);

      copy_edge2d(b[i], d[i]);
      addeval_edge2d(-1.0, -1.0, x[i], d[i]);

      oldnorm = norm;
      norm = norm2_edge2d(d[i]);
      oldnorm = l2norm;
      l2norm = l2norm_edge2d(x[i], example2_sol, 0);
#pragma omp single
      printf("Level %2d: residual %.2e, L^2 error %.2e (%.2f)\n",
	     i, norm, l2norm, l2norm/oldnorm);
    }    

    l2norm = l2norm_edge2d(x[L], example2_sol, 0);
    interpolate_edge2d(d[L], example2_sol, 0);
    norm = l2norm_edge2d(d[L], example2_sol, 0);
#pragma omp single
    printf("Final error %.4e, interpolation error %.4e\n",
	   l2norm, norm);

    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("%.2f seconds, %.2f seconds per iteration\n",
	   t_run, t_run / 10);
  }
    
  return 0;
}

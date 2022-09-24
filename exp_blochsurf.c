
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdarg.h>

#include "edge2d.h"
#include "node2d.h"
#include "basic.h"
#include "linalg.h"
#include "parameters.h"

#ifdef USE_CAIRO
#include <cairo.h>
#endif

/* ------------------------------------------------------------
 * Set up the Bloch boundary conditions
 * ------------------------------------------------------------ */

static void
blochsurface(int ix, int iy, int blochsteps,
	     real *xbloch, real *ybloch,
	     field *xfactor, field *yfactor)
{
  real xk, yk;

  xk = 2.0 * M_PI * ix / (blochsteps - 1) - M_PI;
  yk = 2.0 * M_PI * iy / (blochsteps - 1) - M_PI;

  *xfactor = EXP(I * xk);
  *yfactor = EXP(I * yk);
  *xbloch = xk;
  *ybloch = yk;
}

/* ------------------------------------------------------------
 * Set up a simple permittivity pattern for an empty lattice
 * ------------------------------------------------------------ */

static epspattern *
new_empty_epspattern()
{
  epspattern *pat;

  pat = new_epspattern(1.0, 0, 0);

  return pat;
}

/* ------------------------------------------------------------
 * Set up a simple permittivity pattern containing a centered circle
 * ------------------------------------------------------------ */

static epspattern *
new_circle_epspattern()
{
  const real eps_domain = 1.0;
  const real eps_circle = 4.0;
  epspattern *pat;

  pat = new_epspattern(eps_domain, 0, 1);

  pat->circle[0] = eps_circle;
  pat->circle[1] = 0.5;
  pat->circle[2] = 0.5;
  pat->circle[3] = 0.333;

  return pat;
}

/* ------------------------------------------------------------
 * Set up Kivshar's pattern, a circle of radius 0.18 centered
 * in a unit square, with an increased permittivity of 11.56
 * ------------------------------------------------------------ */

static epspattern *
new_kivshar_epspattern()
{
  epspattern *pat;

  pat = new_epspattern(1.0, 0, 1);

  pat->circle[0] = 11.56;
  pat->circle[1] = 0.5;
  pat->circle[2] = 0.5;
  pat->circle[3] = 0.18;

  return pat;
}

/* ------------------------------------------------------------
 * Set up a permittivity pattern with four circles set around
 * the center
 * ------------------------------------------------------------ */

static epspattern *
new_centeredcircles_epspattern(real r1, real d1, real r2, real d2,
			       real r3, real d3, real r4, real d4)
{
  const real eps_domain = 1.0;
  const real eps_circle = 4.0;
  epspattern *pat;

  pat = new_epspattern(eps_domain, 0, 4);

  pat->circle[0] = eps_circle;
  pat->circle[1] = 0.5 - sqrt(0.5 * d1 * d1);
  pat->circle[2] = 0.5 - sqrt(0.5 * d1 * d1);
  pat->circle[3] = r1;

  pat->circle[4] = eps_circle;
  pat->circle[5] = 0.5 + sqrt(0.5 * d2 * d2);
  pat->circle[6] = 0.5 - sqrt(0.5 * d2 * d2);
  pat->circle[7] = r2;

  pat->circle[8] = eps_circle;
  pat->circle[9] = 0.5 - sqrt(0.5 * d3 * d3);
  pat->circle[10] = 0.5 + sqrt(0.5 * d3 * d3);
  pat->circle[11] = r3;

  pat->circle[12] = eps_circle;
  pat->circle[13] = 0.5 + sqrt(0.5 * d4 * d4);
  pat->circle[14] = 0.5 + sqrt(0.5 * d4 * d4);
  pat->circle[15] = r4;

  return pat;
}

/* ------------------------------------------------------------
 * Perform one step of the block preconditioned inverse iteration
 * ------------------------------------------------------------ */

static void
pinvit_step(int l, int eigs,
	    int smoother_steps, int prec_steps, int gradient_steps,
	    real *lambda, real *norm, real *oldnorm,
	    edge2d **e, edge2d **u,
	    const matrix *Ae, edge2d **x, edge2d **b, edge2d **d,
	    const matrix *An, node2d **xg, node2d **bg, node2d **dg)
{
  edge2d *aux;
  matrix *Ar, *Mr;
  field alpha, beta;
  int rank = omp_get_thread_num();
  real newnorm;
  static field val;
  int i, j, info;

#pragma omp barrier
  
  /* Perform one PINVIT step for every eigenvector approximation */
  for(i=0; i<eigs; i++)
    pinvit_edge2d(l, 2, prec_steps, gradient_steps,
		  lambda[i], e[i],
		  Ae, b, x, d, An, bg, xg, dg);

#pragma omp barrier

  if(rank == 0) {
    /* Switch old and new approximations */
    for(i=0; i<eigs; i++) {
      aux = u[i];
      u[i] = e[i];
      e[i] = aux;
    }

    /* Set up Ritz system */
    Ar = new_matrix(eigs, eigs);
    Mr = new_matrix(eigs, eigs);
  }

#pragma omp barrier
  
  for(j=0; j<eigs; j++)
    for(i=j; i<eigs; i++) {
      alpha = energyprod_edge2d(1.0, 1.0, u[i], u[j]);
      beta = massprod_edge2d(u[i], u[j]);

      if(rank == 0) {
	Ar->a[i+j*Ar->ld] = alpha;
	Mr->a[i+j*Mr->ld] = beta;
      }
    }

  /* Solve Ritz system */
  if(rank == 0) {
    info = sygv_matrix(Ar, Mr, lambda);
    assert(info == 0);
  }

  /* Construct eigenvector approximations */
  for(j=0; j<eigs; j++) {
    zero_edge2d(e[j]);
    for(i=0; i<eigs; i++) {
      if(rank == 0)
	val = Ar->a[i+j*Ar->ld];

#pragma omp barrier

      add_edge2d(val, u[i], e[j]);
    }
  }

  /* Clean up */
  if(rank == 0) {
    del_matrix(Mr);
    del_matrix(Ar);
  }

  /* Compute defects */
  for(i=0; i<eigs; i++) {
    zero_edge2d(d[l]);
    addeval_edge2d(1.0, 1.0-lambda[i], e[i], d[l]);
    newnorm = norm2_edge2d(d[l]);
    
    if(rank == 0) {
      oldnorm[i] = norm[i];
      norm[i] = newnorm;
    }
  }

#pragma omp barrier
}

/* ------------------------------------------------------------
 * Report on the initial approximation
 * ------------------------------------------------------------ */

static real
initial_report(int eigs, int ieigs,
	       const real *lambda, const real *norm, const real *dnorm)
{
  real maxnorm;
  int i;
  
#pragma omp barrier
  
  maxnorm = norm[0];
  for(i=1; i<ieigs; i++)
    if(norm[i] > maxnorm)
      maxnorm = norm[i];
    
#pragma omp single
  {
    printf("Ritz values:");
    for(i=0; i<ieigs; i++)
      printf(" %f", lambda[i] - 1.0);
    printf("  ");
    for(; i<eigs; i++)
      printf(" %f", lambda[i] - 1.0);
    printf("\n");
    
    printf(" Defects:");
    for(i=0; i<ieigs; i++)
      printf(" %.2e", norm[i]);
    printf("  ");
    for(; i<eigs; i++)
      printf(" %.2e", norm[i]);
    printf("\n");
    
    printf("   Duals:");
    for(i=0; i<ieigs; i++)
      printf(" %.2e", dnorm[i]);
    printf("  ");
    for(; i<eigs; i++)
      printf(" %.2e", dnorm[i]);
    printf("\n");
    
    fflush(stdout);
  }

  return maxnorm;
}

/* ------------------------------------------------------------
 * Report on one step of the iteration
 * ------------------------------------------------------------ */

static real
pinvit_report(int eigs, int ieigs,
	      const real *lambda, const real *norm, const real *oldnorm)
{
  real maxnorm;
  int i;

#pragma omp barrier
  
  maxnorm = norm[0];
  for(i=1; i<ieigs; i++)
    if(norm[i] > maxnorm)
      maxnorm = norm[i];
    
#pragma omp single
  {
    for(i=0; i<ieigs; i++)
      printf(" %f", lambda[i] - 1.0);
    printf("  ");
    for(; i<eigs; i++)
      printf(" %f", lambda[i] - 1.0);
    printf("\n");
    
    printf(" Defects:");
    for(i=0; i<ieigs; i++)
      printf(" %.2e", norm[i]);
    printf("  ");
    for(; i<eigs; i++)
      printf(" %.2e", norm[i]);
    printf("\n");
    
    printf("   Rates:");
    for(i=0; i<ieigs; i++)
      printf(" %.3f", norm[i]/oldnorm[i]);
    printf("  ");
    for(; i<eigs; i++)
      printf(" %.3f", norm[i]/oldnorm[i]);
    printf("\n");
    
    fflush(stdout);
  }

  return maxnorm;
}

/* ------------------------------------------------------------
 * Rayleigh-Ritz extrapolation of eigenvectors
 * ------------------------------------------------------------ */

static void
rayleigh_extrapolation(int sources, int eigs, int ieigs,
		       int worksize, edge2d **work, edge2d **e, ...)
{
  va_list src_list;
  edge2d **s;
  static field *tau = 0;
  static matrix *Ar = 0;
  real *lambdar;
  int dimension = sources * ieigs;
  int rank = omp_get_thread_num();
  int info;
  int i, j;

  assert(eigs <= dimension);
  assert(dimension + 1 <= worksize);

  lambdar = 0;
  
  /* Construct basis from provided eigenvector approximations */
  if(rank == 0) {
    tau = (field *) malloc(sizeof(field) * dimension);
    Ar = new_matrix(dimension, dimension);
    lambdar = (real *) malloc(sizeof(real) * dimension);
  }
    
  va_start(src_list, e);

  for(i=0; i<sources; i++) {
    s = va_arg(src_list, edge2d **);
    for(j=0; j<ieigs; j++)
      copy_edge2d(s[j], work[j+i*ieigs]);
  }

  va_end(src_list);
  
#pragma omp barrier

  /* M-orthogonalize basis */
  orthonormalize_edge2d(dimension, work, tau, work[dimension]);
  
  /* Set up Ritz system */
  for(j=0; j<dimension; j++)
    for(i=j; i<dimension; i++)
      Ar->a[i+j*Ar->ld] = energyprod_edge2d(1.0, 1.0, work[i], work[j]);
  
  /* Solve Ritz system */
  if(rank == 0) {
    info = syev_matrix(Ar, lambdar);
    assert(info == 0);
  }

#pragma omp barrier
  
  /* Construct eigenvector approximations */
  for(j=0; j<eigs; j++) {
    zero_edge2d(e[j]);
    for(i=0; i<dimension; i++)
      add_edge2d(Ar->a[i+j*Ar->ld], work[i], e[j]);
  }

  /* Clean up */
  if(rank == 0) {
    free(lambdar);
    lambdar = 0;
    
    del_matrix(Ar);
    Ar = 0;
    
    free(tau);
    tau = 0;
  }

#pragma omp barrier
}

/* ============================================================
 * Main program
 * ============================================================ */

int
main(int argc, char **argv)
{
  grid2d **gr;
  epspattern *pat;
  edge2d ***e, ***u;
  edge2d **x, **b, **d;
  edge2d ***yext, ***xext;
  int rayleighsize;
  edge2d **rayleighwork;
  node2d **xg, **bg, **dg;
  matrix *A, *M, *Ae, *An;
  real **lambda;
  field product1, product2;
  real *oldnorm, *norm, *dnorm;
  real maxnorm;
  real t_start, t_stop, t_run;
  char filename[80];
  char gridfilename[80];
  FILE *out;
  size_t sz;
  int blochsteps;
  char permittivity;
  real tolerance;
  int nahidpattern;
  int pinvit_steps = 5;
  int prec_steps = 2;
  int gradient_steps = 5;
  int bloch_steps = 10;
  int n;
  int L, eigs, ieigs, throwaway;
  int ev;
  int i, j, k, l, ix, iy;

  printf("========================================\n"
	 "Eigenvalues with Bloch boundary conditions\n"
	 "========================================\n");

  n = askforint("Coarse grid intervals?", "maxwell_coarse", 8);

  L = askforint("Refinement levels?", "maxwell_levels", 7);

  permittivity = askforchar("Permittivity pattern? (E)mpty, (C)ircle, (K)ivshar, or (N)ahid?",
			    "maxwell_permittivity", "eckn", 'e');

  nahidpattern = (permittivity == 'n' ?
		  askforint("Which of Nahid's patterns? 1 to 6 is possible.",
			    "maxwell_pattern", 2) : 2);

  ieigs = askforint("Number of eigenvalues?",
		    "maxwell_eigs", 16);

  throwaway = askforint("Number of throw-away eigenvalues?",
			"maxwell_throwaway", 8);

  eigs = ieigs + throwaway;
  
  blochsteps = askforint("Number of Bloch steps per edge?",
			 "maxwell_blochsteps", 20);
  
  strncpy(filename, "blochsurf.dat", 80);
  askforstring("Protocol file name?", "maxwell_protocol", filename, 80);

  strncpy(gridfilename, "blochsurf.png", 80);
  askforstring("Grid file name?", "maxwell_grid", gridfilename, 80);

  pinvit_steps = 5;
  bloch_steps = 8;
  tolerance = 0.0;

  pinvit_steps = askforint("Maximal steps for nested iteration?",
			   "maxwell_pinvit", 40);
  bloch_steps = askforint("Maximal steps for parameter tracing?",
			  "maxwell_bloch", 40);
  tolerance = askforreal("Tolerance for the residual?", "maxwell_residual",
			 1e-2);

  out = fopen(filename, "w");

  fprintf(out,
	  "# Bloch eigenvalues\n"
	  "# Coarse grid intervals: %d\n"
	  "# Refinement levels: %d\n"
	  "# %d eigenvalues computed, plus %d throwaway eigenvalues\n"
	  "\n"
	  "# 1: Bloch x index, 0 to %d\n"
	  "# 2: Bloch y index, 0 to %d\n"
	  "# 3: First eigenvalue\n"
	  "# %d: Last eigenvalue\n"
	  "# %d: Number of iteration steps\n"
	  "\n",
	  n, L, ieigs, throwaway,
	  blochsteps-1, blochsteps-1,
	  eigs+2, eigs+3);
  
  gr = (grid2d **) malloc(sizeof(grid2d *) * (L+1));
  e = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  u = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  yext = (edge2d ***) malloc(sizeof(edge2d **) * 3 * 3);
  xext = (edge2d ***) malloc(sizeof(edge2d **) * 3);
  x = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  b = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  d = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  xg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  bg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  dg = (node2d **) malloc(sizeof(node2d *) * (L+1));
  lambda = (real **) malloc(sizeof(real *) * (L+1));
  
  gr[L] = new_grid2d(n << L, n << L, 1.0 / (n << L), 1.0 / (n << L));

  switch(permittivity) {
  default:
    /* Falls through */
  case 'e':
    printf("Permittivity pattern: Empty lattice\n");
    pat = new_empty_epspattern();
    break;

  case 'c':
    printf("Permittivity pattern: Circle\n");
    pat = new_circle_epspattern();
    break;

  case 'k':
    printf("Permittivity pattern: Kivshar\n");
    pat = new_kivshar_epspattern();
    break;
    
  case 'n':
    printf("Permittivity pattern: Nahid's special pattern #%d\n",
	   nahidpattern);
    switch(nahidpattern) {
    default:
      /* Falls through */
    case 1:
      pat = new_centeredcircles_epspattern(0.125, sqrt(0.125),
					   0.125, sqrt(0.125),
					   0.125, sqrt(0.125),
					   0.125, sqrt(0.125));
      break;

    case 2:
      pat = new_centeredcircles_epspattern(0.2, sqrt(0.125),
					   0.1, sqrt(0.125),
					   0.1, sqrt(0.125),
					   0.2, sqrt(0.125));
      break;
    
    case 3:
      pat = new_centeredcircles_epspattern(0.125, sqrt(0.125),
					   0.125, sqrt(0.125),
					   0.125, sqrt(0.125),
					   0.125, sqrt(2.0 * 0.135*0.135));
      break;

    case 4:
      pat = new_centeredcircles_epspattern(0.125, sqrt(0.03645),
					   0.125, sqrt(0.125),
					   0.125, sqrt(0.125),
					   0.125, sqrt(0.03645));
      break;

    case 5:
      pat = new_centeredcircles_epspattern(0.2, sqrt(0.125),
					   0.1, sqrt(0.125),
					   0.1, sqrt(0.125),
					   0.2, sqrt(0.0512));
      break;

    case 6:
      pat = new_centeredcircles_epspattern(0.2, sqrt(0.0512),
					   0.1, sqrt(0.125),
					   0.1, sqrt(0.125),
					   0.2, sqrt(0.0512));
      break;
    }
    break;
  }
  setpattern_grid2d(pat, gr[L]);

  blochsurface(0, 0, blochsteps,
	       &gr[L]->xbloch, &gr[L]->ybloch,
	       &gr[L]->xfactor, &gr[L]->yfactor);

#ifdef USE_CAIRO
  draw2d_grid2d(gr[L], gridfilename);
#endif
  
  for(l=L; l-->0; ) {
    gr[l] = coarsen_grid2d(gr[l+1]);

#ifdef USE_CAIRO
    if(l <= 5) {
      snprintf(filename, 80, "Pictures/grid%02d.png", l);
      draw2d_grid2d(gr[l], filename);
    }
#endif
  }
  
  for(l=0; l<=L; l++) {
    e[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      e[l][i] = new_edge2d(gr[l]);
    u[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      u[l][i] = new_edge2d(gr[l]);
    x[l] = new_edge2d(gr[l]);
    b[l] = new_edge2d(gr[l]);
    d[l] = new_edge2d(gr[l]);
    xg[l] = new_node2d(gr[l]);
    bg[l] = new_node2d(gr[l]);
    dg[l] = new_node2d(gr[l]);
  }

  for(j=0; j<9; j++) {
    yext[j] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      yext[j][i] = new_edge2d(gr[L]);
  }
  for(j=0; j<3; j++) {
    xext[j] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      xext[j][i] = new_edge2d(gr[L]);
  }

  rayleighsize = 3 * eigs + 1;
  rayleighwork = (edge2d **) malloc(sizeof(edge2d *) * rayleighsize);
  for(i=0; i<rayleighsize; i++)
    rayleighwork[i] = new_edge2d(gr[L]);
  
  lambda[0] = (real *) malloc(sizeof(real) * 2 * n * n);
  for(l=1; l<=L; l++)
    lambda[l] = (real *) malloc(sizeof(real) * eigs);

  norm = (real *) malloc(sizeof(real) * eigs);
  oldnorm = (real *) malloc(sizeof(real) * eigs);
  dnorm = (real *) malloc(sizeof(real) * eigs);
  
  for(l=0; l<=L; l++) {
    sz = (sizeof(field) * gr[l]->nx * (gr[l]->ny+1)
	  + sizeof(field) * (gr[l]->nx+1) * gr[l]->ny
	  + sizeof(edge2d));
    printf("Level %2d: %d x %d grid, %zd dofs, %.1f MB per gridfunc\n",
	   l, gr[l]->nx, gr[l]->ny,
	   getdimension_edge2d(x[l]),
	   sz / 1048576.0);
  }

  printf("Setting up nodal coarse-grid solver\n");
  An = new_matrix(n*n, n*n);
  densematrix_node2d(true, gr[0], An);
  potrf_matrix(An);

  printf("Setting up edge coarse-grid solver\n");
  Ae = new_matrix(2*n*n, 2*n*n);
  densematrix_edge2d(1.0, 1.0, gr[0], Ae);
  potrf_matrix(Ae);
  
  printf("Setting up coarse-grid stiffness and mass matrix\n");
  t_start = omp_get_wtime();
  A = new_matrix(2*n*n, 2*n*n);
  M = new_matrix(2*n*n, 2*n*n);
  densematrix_edge2d(1.0, 0.0, gr[0], A);
  densematrix_edge2d(0.0, 1.0, gr[0], M);
  t_stop = omp_get_wtime();
  t_run = t_stop - t_start;
  printf("  %.2f seconds\n", t_run);
  
  printf("Solving coarse-grid eigenvalue problem\n");
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

#pragma omp parallel private(i,j,ix,iy,k,l,t_start,t_stop,t_run,product1,product2)
  {
    /* ----------------------------------------
     * Nested iteration
     * ---------------------------------------- */

    k = 0;
    
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
	vcycle_edge2d(l, 2, Ae, b, x, d);
	dnorm[i] = sqrt(fabs(dotprod_edge2d(x[l], b[l]))) / norm[i];
      }

      /* Report on initial approximation */
#pragma omp single
      printf("----------------------------------------\n"
	     "Level %d:\n", l);
      maxnorm = initial_report(eigs, ieigs, lambda[l], norm, dnorm);

      for(k=0; k<pinvit_steps && maxnorm>tolerance; k++) {
	/* Perform preconditioned inverse iteration for all
	 * current eigenvector approximations */
	pinvit_step(l, eigs,
		    2, prec_steps, gradient_steps,
		    lambda[l], norm, oldnorm,
		    e[l], u[l], Ae, x, b, d, An, xg, bg, dg);

#pragma omp single
	printf(" Step %2d:", k+1);
	maxnorm = pinvit_report(eigs, ieigs, lambda[l], norm, oldnorm);
      }

      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for level %d\n", t_run, l);
    }

    /* Store in y extrapolation structure */
    for(i=0; i<eigs; i++)
      copy_edge2d(e[L][i], yext[0][i]);

#pragma omp single
    {
      fprintf(out, "0 0");
      for(i=0; i<eigs; i++)
	fprintf(out, " %f", lambda[L][i] - 1.0);
      fprintf(out, "  %d\n", k);
      fflush(out);
    }

    /* ----------------------------------------
     * Bloch scan of the first row
     * ---------------------------------------- */

#pragma omp single
    {
      printf("----------------------------------------\n"
	     "Point (1,0)\n");
	
      /* Set new Bloch parameters */
      blochsurface(1, 0, blochsteps,
		   &gr[L]->xbloch, &gr[L]->ybloch,
		   &gr[L]->xfactor, &gr[L]->yfactor);
      for(i=0; i<L; i++) {
	gr[i]->xbloch = gr[L]->xbloch;
	gr[i]->ybloch = gr[L]->ybloch;
	gr[i]->xfactor = gr[L]->xfactor;
	gr[i]->yfactor = gr[L]->yfactor;
      }

      /* Recompute edge coarse-grid matrix */
      del_matrix(Ae);
      Ae = new_matrix(2*n*n, 2*n*n);
      densematrix_edge2d(1.0, 1.0, gr[0], Ae);
      potrf_matrix(Ae);
	
      /* Recompute nodal coarse-grid matrix */
      del_matrix(An);
      An = new_matrix(n*n, n*n);
      densematrix_node2d(true, gr[0], An);
      potrf_matrix(An);
    }

    t_start = omp_get_wtime();

    for(i=0; i<eigs; i++) {
      /* Compute A e */
      zero_edge2d(b[L]);
      addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	
      /* Compute M e */
      zero_edge2d(d[L]);
      addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	
      /* Compute the Rayleigh quotient */
      product1 = dotprod_edge2d(e[L][i], b[L]);
      product2 = dotprod_edge2d(e[L][i], d[L]);
      lambda[L][i] = product1 / product2;

      /* Compute the defect A e - lambda M e */
      add_edge2d(-lambda[L][i], d[L], b[L]);
	
      norm[i] = norm2_edge2d(b[L]);
	
      /* Estimate the dual norm */
      zero_edge2d(x[L]);
      vcycle_edge2d(L, 2, Ae, b, x, d);
      dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
    }

    maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);

    for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
      pinvit_step(L, eigs,
		  2, prec_steps, gradient_steps,
		  lambda[L], norm, oldnorm,
		  e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	
#pragma omp single
      printf(" Step %2d:", k+1);

      maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
    }
      
    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("  %.2f seconds for point (1,0)\n", t_run);

    /* Store in y extrapolation structure */
    for(i=0; i<eigs; i++)
      copy_edge2d(e[L][i], yext[1][i]);

#pragma omp single
    {
      fprintf(out, "1 0");
      for(i=0; i<eigs; i++)
	fprintf(out, " %f", lambda[L][i] - 1.0);
      fprintf(out, "  %d\n", k);
      fflush(out);
    }

#pragma omp single
    {
      printf("----------------------------------------\n"
	     "Point (2,0)\n");

      /* Set new Bloch parameters */
      blochsurface(2, 0, blochsteps,
		   &gr[L]->xbloch, &gr[L]->ybloch,
		   &gr[L]->xfactor, &gr[L]->yfactor);
      for(i=0; i<L; i++) {
	gr[i]->xbloch = gr[L]->xbloch;
	gr[i]->ybloch = gr[L]->ybloch;
	gr[i]->xfactor = gr[L]->xfactor;
	gr[i]->yfactor = gr[L]->yfactor;
      }

      /* Recompute edge coarse-grid matrix */
      del_matrix(Ae);
      Ae = new_matrix(2*n*n, 2*n*n);
      densematrix_edge2d(1.0, 1.0, gr[0], Ae);
      potrf_matrix(Ae);
	
      /* Recompute nodal coarse-grid matrix */
      del_matrix(An);
      An = new_matrix(n*n, n*n);
      densematrix_node2d(true, gr[0], An);
      potrf_matrix(An);
    }

    t_start = omp_get_wtime();
      
    /* Rayleigh extrapolation using (1,0) and (0,0) */
    rayleigh_extrapolation(2, eigs, ieigs,
			   rayleighsize, rayleighwork, e[L],
			   yext[0], yext[1]);

    for(i=0; i<eigs; i++) {
      /* Compute A e */
      zero_edge2d(b[L]);
      addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	
      /* Compute M e */
      zero_edge2d(d[L]);
      addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	
      /* Compute the Rayleigh quotient */
      product1 = dotprod_edge2d(e[L][i], b[L]);
      product2 = dotprod_edge2d(e[L][i], d[L]);
      lambda[L][i] = product1 / product2;

      /* Compute the defect A e - lambda M e */
      add_edge2d(-lambda[L][i], d[L], b[L]);
	
      norm[i] = norm2_edge2d(b[L]);
	
      /* Estimate the dual norm */
      zero_edge2d(x[L]);
      vcycle_edge2d(L, 2, Ae, b, x, d);
      dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
    }

    maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);

    for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
      pinvit_step(L, eigs,
		  2, prec_steps, gradient_steps,
		  lambda[L], norm, oldnorm,
		  e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	
#pragma omp single
      printf(" Step %2d:", k+1);

      maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
    }
      
    t_stop = omp_get_wtime();
    t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
    printf("  %.2f seconds for point (2,0)\n", t_run);

    /* Store in y extrapolation structure */
    for(i=0; i<eigs; i++)
      copy_edge2d(e[L][i], yext[2][i]);
      
#pragma omp single
    {
      fprintf(out, "2 0");
      for(i=0; i<eigs; i++)
	fprintf(out, " %f", lambda[L][i] - 1.0);
      fprintf(out, "  %d\n", k);
      fflush(out);
    }

    /* Initialize x extrapolation structure */
    for(j=0; j<3; j++)
      for(i=0; i<eigs; i++)
	copy_edge2d(yext[j][i], xext[j][i]);
    
    for(ix=3; ix<blochsteps; ix++) {
#pragma omp single
      {
	printf("----------------------------------------\n"
	       "Point (%d,0)\n", ix);

	/* Set new Bloch parameters */
	blochsurface(ix, 0, blochsteps,
		     &gr[L]->xbloch, &gr[L]->ybloch,
		     &gr[L]->xfactor, &gr[L]->yfactor);
	for(i=0; i<L; i++) {
	  gr[i]->xbloch = gr[L]->xbloch;
	  gr[i]->ybloch = gr[L]->ybloch;
	  gr[i]->xfactor = gr[L]->xfactor;
	  gr[i]->yfactor = gr[L]->yfactor;
	}

	/* Recompute edge coarse-grid matrix */
	del_matrix(Ae);
	Ae = new_matrix(2*n*n, 2*n*n);
	densematrix_edge2d(1.0, 1.0, gr[0], Ae);
	potrf_matrix(Ae);
	
	/* Recompute nodal coarse-grid matrix */
	del_matrix(An);
	An = new_matrix(n*n, n*n);
	densematrix_node2d(true, gr[0], An);
	potrf_matrix(An);
      }

      t_start = omp_get_wtime();
	
      /* Rayleigh extrapolation using (ix-1,0), (ix-2,0), and (ix-3,0) */
      rayleigh_extrapolation(3, eigs, ieigs,
			     rayleighsize, rayleighwork, e[L],
			     xext[ix%3], xext[(ix+1)%3], xext[(ix+2)%3]);

      for(i=0; i<eigs; i++) {
	/* Compute A e */
	zero_edge2d(b[L]);
	addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	  
	/* Compute M e */
	zero_edge2d(d[L]);
	addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	  
	/* Compute the Rayleigh quotient */
	product1 = dotprod_edge2d(e[L][i], b[L]);
	product2 = dotprod_edge2d(e[L][i], d[L]);
	lambda[L][i] = product1 / product2;
	  
	/* Compute the defect A e - lambda M e */
	add_edge2d(-lambda[L][i], d[L], b[L]);
	  
	norm[i] = norm2_edge2d(b[L]);
	  
	/* Estimate the dual norm */
	zero_edge2d(x[L]);
	vcycle_edge2d(L, 2, Ae, b, x, d);
	dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
      }
	
      maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);

      for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
	pinvit_step(L, eigs,
		    2, prec_steps, gradient_steps,
		    lambda[L], norm, oldnorm,
		    e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	
#pragma omp single
	printf(" Step %2d:", k+1);

	maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
      }
	
      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for point (%d,0)\n", t_run, ix);

      /* Store in x extrapolation structure */
      for(i=0; i<eigs; i++)
	copy_edge2d(e[L][i], xext[ix%3][i]);
      
#pragma omp single
      {
	fprintf(out, "%d 0", ix);
	for(i=0; i<eigs; i++)
	  fprintf(out, " %f", lambda[L][i] - 1.0);
	fprintf(out, "  %d\n", k);
	fflush(out);
      }
    }

#pragma omp single
    fprintf(out, "\n");
    
    /* ----------------------------------------
     * Bloch scan of the remaining rows
     * ---------------------------------------- */

    for(iy=1; iy<blochsteps; iy++) {
#pragma omp single
      {
	printf("----------------------------------------\n"
	       "Point (0,%d)\n", iy);

	/* Set new Bloch parameters */
	blochsurface(0, iy, blochsteps,
		     &gr[L]->xbloch, &gr[L]->ybloch,
		     &gr[L]->xfactor, &gr[L]->yfactor);
	for(i=0; i<L; i++) {
	  gr[i]->xbloch = gr[L]->xbloch;
	  gr[i]->ybloch = gr[L]->ybloch;
	  gr[i]->xfactor = gr[L]->xfactor;
	  gr[i]->yfactor = gr[L]->yfactor;
	}

	/* Recompute edge coarse-grid matrix */
	del_matrix(Ae);
	Ae = new_matrix(2*n*n, 2*n*n);
	densematrix_edge2d(1.0, 1.0, gr[0], Ae);
	potrf_matrix(Ae);
	  
	/* Recompute nodal coarse-grid matrix */
	del_matrix(An);
	An = new_matrix(n*n, n*n);
	densematrix_node2d(true, gr[0], An);
	potrf_matrix(An);
      }
	
      t_start = omp_get_wtime();

      if(iy == 1) {
	/* Use (0,0) as initial guess */
	for(i=0; i<eigs; i++)
	  copy_edge2d(yext[0][i], e[L][i]);
      }
      else if(iy == 2) {
	/* Rayleigh extrapolation using (0,0) and (0,1) */
	rayleigh_extrapolation(2, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       yext[0], yext[3]);
      }
      else {
	/* Rayleigh extrapolation using (0,iy-1), (0,iy-2), and (0,iy-3) */
	rayleigh_extrapolation(3, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       yext[3*(iy%3)],
			       yext[3*((iy+1)%3)],
			       yext[3*((iy+2)%3)]);
      }	

      for(i=0; i<eigs; i++) {
	/* Compute A e */
	zero_edge2d(b[L]);
	addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	  
	/* Compute M e */
	zero_edge2d(d[L]);
	addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	  
	/* Compute the Rayleigh quotient */
	product1 = dotprod_edge2d(e[L][i], b[L]);
	product2 = dotprod_edge2d(e[L][i], d[L]);
	lambda[L][i] = product1 / product2;
	  
	/* Compute the defect A e - lambda M e */
	add_edge2d(-lambda[L][i], d[L], b[L]);
	  
	norm[i] = norm2_edge2d(b[L]);
	  
	/* Estimate the dual norm */
	zero_edge2d(x[L]);
	vcycle_edge2d(L, 2, Ae, b, x, d);
	dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
      }
	
      maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);

      for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
	pinvit_step(L, eigs,
		    2, prec_steps, gradient_steps,
		    lambda[L], norm, oldnorm,
		    e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	  
#pragma omp single
	printf(" Step %2d:", k+1);
	  
	maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
      }
	
      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for point (0,%d)\n", t_run, iy);

      /* Store y in extrapolation structure */
      for(i=0; i<eigs; i++)
	copy_edge2d(e[L][i], yext[3*(iy%3)][i]);

#pragma omp single
      {
	fprintf(out, "0 %d", iy);
	for(i=0; i<eigs; i++)
	  fprintf(out, " %f", lambda[L][i] - 1.0);
	fprintf(out, "  %d\n", k);
	fflush(out);
      }

#pragma omp single
      {
	printf("----------------------------------------\n"
	       "Point (1,%d)\n", iy);

	/* Set new Bloch parameters */
	blochsurface(1, iy, blochsteps,
		     &gr[L]->xbloch, &gr[L]->ybloch,
		     &gr[L]->xfactor, &gr[L]->yfactor);
	for(i=0; i<L; i++) {
	  gr[i]->xbloch = gr[L]->xbloch;
	  gr[i]->ybloch = gr[L]->ybloch;
	  gr[i]->xfactor = gr[L]->xfactor;
	  gr[i]->yfactor = gr[L]->yfactor;
	}
	
	/* Recompute edge coarse-grid matrix */
	del_matrix(Ae);
	Ae = new_matrix(2*n*n, 2*n*n);
	densematrix_edge2d(1.0, 1.0, gr[0], Ae);
	potrf_matrix(Ae);
	
	/* Recompute nodal coarse-grid matrix */
	del_matrix(An);
	An = new_matrix(n*n, n*n);
	densematrix_node2d(true, gr[0], An);
	potrf_matrix(An);
      }
	
      t_start = omp_get_wtime();
	
      if(iy == 1) {
	/* Use (1,0) as initial guess */
	for(i=0; i<eigs; i++)
	  copy_edge2d(yext[1][i], e[L][i]);
      }
      else if(iy == 2) {
	/* Rayleigh extrapolation using (1,0) and (1,1) */
	rayleigh_extrapolation(2, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       yext[1], yext[4]);
      }
      else {
	/* Rayleigh extrapolation using (1,iy-1), (1,iy-2), and (1,iy-3) */
	rayleigh_extrapolation(3, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       yext[1+3*(iy%3)],
			       yext[1+3*((iy+1)%3)],
			       yext[1+3*((iy+2)%3)]);
      }	

      for(i=0; i<eigs; i++) {
	/* Compute A e */
	zero_edge2d(b[L]);
	addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	  
	/* Compute M e */
	zero_edge2d(d[L]);
	addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	  
	/* Compute the Rayleigh quotient */
	product1 = dotprod_edge2d(e[L][i], b[L]);
	product2 = dotprod_edge2d(e[L][i], d[L]);
	lambda[L][i] = product1 / product2;
	  
	/* Compute the defect A e - lambda M e */
	add_edge2d(-lambda[L][i], d[L], b[L]);
	  
	norm[i] = norm2_edge2d(b[L]);
	  
	/* Estimate the dual norm */
	zero_edge2d(x[L]);
	vcycle_edge2d(L, 2, Ae, b, x, d);
	dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
      }
	
      maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);

      for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
	pinvit_step(L, eigs,
		    2, prec_steps, gradient_steps,
		    lambda[L], norm, oldnorm,
		    e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	  
#pragma omp single
	printf(" Step %2d:", k+1);
	  
	maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
      }
	
      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for point (1,%d)\n", t_run, iy);

      /* Store in y extrapolation structure */
      for(i=0; i<eigs; i++)
	copy_edge2d(e[L][i], yext[1+3*(iy%3)][i]);

#pragma omp single
      {
	fprintf(out, "1 %d", iy);
	for(i=0; i<eigs; i++)
	  fprintf(out, " %f", lambda[L][i] - 1.0);
	fprintf(out, "  %d\n", k);
	fflush(out);
      }

#pragma omp single
      {
	printf("----------------------------------------\n"
	       "Point (2,%d)\n", iy);

	/* Set new Bloch parameters */
	blochsurface(2, iy, blochsteps,
		     &gr[L]->xbloch, &gr[L]->ybloch,
		     &gr[L]->xfactor, &gr[L]->yfactor);
	for(i=0; i<L; i++) {
	  gr[i]->xbloch = gr[L]->xbloch;
	  gr[i]->ybloch = gr[L]->ybloch;
	  gr[i]->xfactor = gr[L]->xfactor;
	  gr[i]->yfactor = gr[L]->yfactor;
	}
	
	/* Recompute edge coarse-grid matrix */
	del_matrix(Ae);
	Ae = new_matrix(2*n*n, 2*n*n);
	densematrix_edge2d(1.0, 1.0, gr[0], Ae);
	potrf_matrix(Ae);
	
	/* Recompute nodal coarse-grid matrix */
	del_matrix(An);
	An = new_matrix(n*n, n*n);
	densematrix_node2d(true, gr[0], An);
	potrf_matrix(An);
      }
	
      t_start = omp_get_wtime();
	
      if(iy == 1) {
	/* Use (2,0) as initial guess */
	for(i=0; i<eigs; i++)
	  copy_edge2d(yext[2][i], e[L][i]);
      }
      else if(iy == 2) {
	/* Rayleigh extrapolation using (2,0) and (2,1) */
	rayleigh_extrapolation(2, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       yext[2], yext[5]);
      }
      else {
	/* Rayleigh extrapolation using (2,iy-1), (2,iy-2), and (2,iy-3) */
	rayleigh_extrapolation(3, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       yext[2+3*(iy%3)],
			       yext[2+3*((iy+1)%3)],
			       yext[2+3*((iy+2)%3)]);
      }	

      for(i=0; i<eigs; i++) {
	/* Compute A e */
	zero_edge2d(b[L]);
	addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	  
	/* Compute M e */
	zero_edge2d(d[L]);
	addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	  
	/* Compute the Rayleigh quotient */
	product1 = dotprod_edge2d(e[L][i], b[L]);
	product2 = dotprod_edge2d(e[L][i], d[L]);
	lambda[L][i] = product1 / product2;
	  
	/* Compute the defect A e - lambda M e */
	add_edge2d(-lambda[L][i], d[L], b[L]);
	  
	norm[i] = norm2_edge2d(b[L]);
	  
	/* Estimate the dual norm */
	zero_edge2d(x[L]);
	vcycle_edge2d(L, 2, Ae, b, x, d);
	dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
      }
	
      maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);

      for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
	pinvit_step(L, eigs,
		    2, prec_steps, gradient_steps,
		    lambda[L], norm, oldnorm,
		    e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	  
#pragma omp single
	printf(" Step %2d:", k+1);
	  
	maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
      }
	
      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for point (2,%d)\n", t_run, iy);

      /* Store in y extrapolation structure */
      for(i=0; i<eigs; i++)
	copy_edge2d(e[L][i], yext[2+3*(iy%3)][i]);

#pragma omp single
      {
	fprintf(out, "2 %d", iy);
	for(i=0; i<eigs; i++)
	  fprintf(out, " %f", lambda[L][i] - 1.0);
	fprintf(out, "  %d\n", k);
	fflush(out);
      }

      /* Initialize x extrapolation structure */
      for(j=0; j<3; j++)
	for(i=0; i<eigs; i++)
	  copy_edge2d(yext[j+3*(iy%3)][i], xext[j][i]);
      
      for(ix=3; ix<blochsteps; ix++) {
#pragma omp single
	{
	  printf("----------------------------------------\n"
		 "Point (%d,%d)\n", ix, iy);

	  /* Set new Bloch parameters */
	  blochsurface(ix, iy, blochsteps,
		       &gr[L]->xbloch, &gr[L]->ybloch,
		       &gr[L]->xfactor, &gr[L]->yfactor);
	  for(i=0; i<L; i++) {
	    gr[i]->xbloch = gr[L]->xbloch;
	    gr[i]->ybloch = gr[L]->ybloch;
	    gr[i]->xfactor = gr[L]->xfactor;
	    gr[i]->yfactor = gr[L]->yfactor;
	  }

	  /* Recompute edge coarse-grid matrix */
	  del_matrix(Ae);
	  Ae = new_matrix(2*n*n, 2*n*n);
	  densematrix_edge2d(1.0, 1.0, gr[0], Ae);
	  potrf_matrix(Ae);
	    
	  /* Recompute nodal coarse-grid matrix */
	  del_matrix(An);
	  An = new_matrix(n*n, n*n);
	  densematrix_node2d(true, gr[0], An);
	  potrf_matrix(An);
	}
      
	t_start = omp_get_wtime();
	  
	/* Rayleigh extrapolation using (ix-1,iy), (ix-2,iy), and (ix-3,iy) */
	rayleigh_extrapolation(3, eigs, ieigs,
			       rayleighsize, rayleighwork, e[L],
			       xext[ix%3], xext[(ix+1)%3], xext[(ix+2)%3]);

	for(i=0; i<eigs; i++) {
	  /* Compute A e */
	  zero_edge2d(b[L]);
	  addeval_edge2d(1.0, 1.0, e[L][i], b[L]);
	    
	  /* Compute M e */
	  zero_edge2d(d[L]);
	  addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	    
	  /* Compute the Rayleigh quotient */
	  product1 = dotprod_edge2d(e[L][i], b[L]);
	  product2 = dotprod_edge2d(e[L][i], d[L]);
	  lambda[L][i] = product1 / product2;
	    
	  /* Compute the defect A e - lambda M e */
	  add_edge2d(-lambda[L][i], d[L], b[L]);
	    
	  norm[i] = norm2_edge2d(b[L]);
	    
	  /* Estimate the dual norm */
	  zero_edge2d(x[L]);
	  vcycle_edge2d(L, 2, Ae, b, x, d);
	  dnorm[i] = sqrt(fabs(dotprod_edge2d(x[L], b[L]))) / norm[i];
	}
	  
	maxnorm = initial_report(eigs, ieigs, lambda[L], norm, dnorm);
	  
	for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
	  pinvit_step(L, eigs,
		      2, prec_steps, gradient_steps,
		      lambda[L], norm, oldnorm,
		      e[L], u[L], Ae, x, b, d, An, xg, bg, dg);
	    
#pragma omp single
	  printf(" Step %2d:", k+1);
	    
	  maxnorm = pinvit_report(eigs, ieigs, lambda[L], norm, oldnorm);
	}
	
	t_stop = omp_get_wtime();
	t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
	printf("  %.2f seconds for point (%d,%d)\n", t_run, ix, iy);

	/* Store in x extrapolation structure */
	for(i=0; i<eigs; i++)
	  copy_edge2d(e[L][i], xext[ix%3][i]);
	  
#pragma omp single
	{
	  fprintf(out, "%d %d", ix, iy);
	  for(i=0; i<eigs; i++)
	    fprintf(out, " %f", lambda[L][i] - 1.0);
	  fprintf(out, "  %d\n", k);
	  fflush(out);
	}
      }
      
#pragma omp single
      fprintf(out, "\n");
    }
    }
  return 0;
}

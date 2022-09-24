
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "edge2d.h"
#include "node2d.h"
#include "basic.h"
#include "linalg.h"
#include "parameters.h"

#ifdef USE_CAIRO
#include <cairo.h>
#endif

static int blochsteps = 50;
static int allblochsteps = 171;

static bool
blochcurve(int pt, real *xbloch, real *ybloch,
	   field *xfactor, field *yfactor)
{
  real xk, yk;

  if(pt < blochsteps) {
    xk = M_PI * pt / blochsteps;
    yk = 0.0;
  }
  else if(pt < 2 * blochsteps) {
    xk = M_PI;
    yk = M_PI * (pt - blochsteps) / blochsteps;
  }
  else {
    xk = M_PI * (1.0 - (pt - 2 * blochsteps) / (1.41 * blochsteps));
    yk = M_PI * (1.0 - (pt - 2 * blochsteps) / (1.41 * blochsteps));
  }

  *xfactor = EXP(I * xk);
  *yfactor = EXP(I * yk);
  *xbloch = xk;
  *ybloch = yk;

  return (pt == blochsteps || pt == 2*blochsteps || pt == allblochsteps-1);
}

static epspattern *
new_circle_epspattern()
{
  epspattern *pat;

  pat = new_epspattern(1.0, 0, 1);

  pat->circle[0] = 4.0;
  pat->circle[1] = 0.5;
  pat->circle[2] = 0.5;
  pat->circle[3] = 0.333;

  return pat;
}

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

static epspattern *
new_nahidmixed_epspattern()
{
  const real eps_domain = 1.0;
  const real eps_circle = 4.0;
  epspattern *pat;
  double scale = 1.0 / 7880;
  
  pat = new_epspattern(eps_domain, 8, 4);

  pat->rectangle[0] = eps_circle;
  pat->rectangle[1] = 1470.0*scale;
  pat->rectangle[2] = 300.0*scale;
  pat->rectangle[3] = 2400.0*scale;
  pat->rectangle[4] = 600.0*scale;

  pat->rectangle[5] = eps_circle;
  pat->rectangle[6] = 1270.0*scale;
  pat->rectangle[7] = 4240.0*scale;
  pat->rectangle[8] = 2400.0*scale;
  pat->rectangle[9] = 600.0*scale;

  pat->rectangle[10] = eps_circle;
  pat->rectangle[11] = 300.0*scale;
  pat->rectangle[12] = 1470.0*scale;
  pat->rectangle[13] = 600.0*scale;
  pat->rectangle[14] = 2400.0*scale;
    
  pat->rectangle[15] = eps_circle;
  pat->rectangle[16] = 4240.0*scale;
  pat->rectangle[17] = 1270.0*scale;
  pat->rectangle[18] = 600.0*scale;
  pat->rectangle[19] = 2400.0*scale;

  pat->rectangle[20] = eps_circle;
  pat->rectangle[21] = 5210.0*scale;
  pat->rectangle[22] = 300.0*scale;
  pat->rectangle[23] = 2400.0*scale;
  pat->rectangle[24] = 600.0*scale;

  pat->rectangle[25] = eps_circle;
  pat->rectangle[26] = 5410.0*scale;
  pat->rectangle[27] = 4240.0*scale;
  pat->rectangle[28] = 2400.0*scale;
  pat->rectangle[29] = 600.0*scale;

  pat->rectangle[30] = eps_circle;
  pat->rectangle[31] = 300.0*scale;
  pat->rectangle[32] = 5210.0*scale;
  pat->rectangle[33] = 600.0*scale;
  pat->rectangle[34] = 2400.0*scale;

  pat->rectangle[35] = eps_circle;
  pat->rectangle[36] = 4240.0*scale;
  pat->rectangle[37] = 5410.0*scale;
  pat->rectangle[38] = 600.0*scale;
  pat->rectangle[39] = 2400.0*scale;
  
  pat->circle[0] = eps_circle;
  pat->circle[1] = 600.0*scale;
  pat->circle[2] = 600.0*scale;
  pat->circle[3] = 600.0*scale;
  
  pat->circle[4] = eps_circle;
  pat->circle[5] = 4540.0*scale;
  pat->circle[6] = 600.0*scale;
  pat->circle[7] = 400.0*scale;
  
  pat->circle[8] = eps_circle;
  pat->circle[9] = 600.0*scale;
  pat->circle[10] = 4540.0*scale;
  pat->circle[11] = 400.0*scale;
  
  pat->circle[12] = eps_circle;
  pat->circle[13] = 4540.0*scale;
  pat->circle[14] = 4540.0*scale;
  pat->circle[15] = 600.0*scale;

  return pat;
}

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

int
main(int argc, char **argv)
{
  grid2d **gr;
  epspattern *pat;
  edge2d ***e, ***u, **eold[2];
  edge2d **x, **b, **d;
  edge2d *aux;
  edge2d **eprev;
  node2d **xg, **bg, **dg;
  matrix *A, *M, *Ae, *An;
  matrix *Ar, *Mr, *Ar2, *Ar3;
  real **lambda, *lambda2, *lambda3;
  real swap;
  field *tau2, *tau3;
  field product1, product2;
  field *chern;
  real *oldnorm, *norm, *dnorm;
  real maxnorm;
  real t_start, t_stop, t_run, t_accum;
  char filename[80];
  char gridfilename[80];
  char cdffilename[80];
  char permittivityname[40];
  FILE *out;
#ifdef USE_NETCDF
  ncfile *cdf;
#endif
  size_t sz;
  char permittivity;
  char stopping;
  char extrapolation;
  char continuity;
  char visualization;
  field alpha;
  field chup;
  real angle, optangle;
  real tolerance;
  int nahidpattern;
  int pinvit_steps = 5;
  int prec_steps = 2;
  int gradient_steps = 5;
  int bloch_steps = 10;
  int chern_vectors;
  int sign;
  int n;
  bool vertex, oldvertex;
  int L, eigs, ieigs, throwaway;
  int ev;
  int i, j, k, l, info;

  printf("========================================\n"
	 "Eigenvalues with Bloch boundary conditions\n"
	 "========================================\n");

  n = askforint("Coarse grid intervals?", "maxwell_coarse", 8);

  L = askforint("Refinement levels?", "maxwell_levels", 7);

  permittivity = askforchar("Permittivity pattern? (E)mpty, (C)ircle, (K)ivshar, (N)ahid?",
			    "maxwell_permittivity", "eckn", 'k');

  nahidpattern = (permittivity == 'n' ?
		  askforint("Which of Nahid's patterns? 0 to 6 is possible.",
			    "maxwell_pattern", 2) : 2);

  ieigs = askforint("Number of eigenvalues?",
		    "maxwell_eigs", 16);

  throwaway = askforint("Number of throw-away eigenvalues?",
			"maxwell_throwaway", 8);

  eigs = ieigs + throwaway;

  chern_vectors = askforint("Number of eigenvectors for the composite Chern number?",
			    "maxwell_chern", 1);

  continuity = askforchar("Eigenvector adjustment strategy? (N)one, (S)ign, (R)eorder?",
			  "maxwell_continuity", "nsr", 's');
  
  if(chern_vectors > ieigs) {
    chern_vectors = ieigs;
    printf("Since only %u eigenvectors are computed reliably, we use %d instead.\n",
	   ieigs, ieigs);
  }
  
  blochsteps = askforint("Number of Bloch steps per edge?",
			 "maxwell_blochsteps", 50);
  allblochsteps = 3.41 * blochsteps + 1;
  
  extrapolation = askforchar("Extrapolation? (N)one, (1)st, or (2)nd order?", "maxwell_extra", "n12", '2');

  strncpy(filename, "bloch2d.dat", 80);
  askforstring("Protocol file name?", "maxwell_protocol", filename, 80);

  strncpy(gridfilename, "grid.png", 80);
  askforstring("Grid file name?", "maxwell_grid", gridfilename, 80);

#ifdef USE_NETCDF
  strncpy(cdffilename, "bloch2d.nc", 80);
  askforstring("CDF file name?", "maxwell_cdf", cdffilename, 80);
#endif
  
  stopping = askforchar("Stopping criterion? (F)ixed iterations or (R)esidual?", "maxwell_stop", "fr", 'r');

  pinvit_steps = 5;
  bloch_steps = 8;
  tolerance = 0.0;

  switch(stopping) {
  default: /* Falls through */
  case 'f':
    pinvit_steps = askforint("Iterations for nested iteration?",
			     "maxwell_pinvit", 5);
    bloch_steps = askforint("Iterations for parameter tracing?",
			    "maxwell_bloch", 8);
    break;

  case 'r':
    tolerance = askforreal("Tolerance for the residual?", "maxwell_residual",
			   1e-2);
    pinvit_steps = askforint("Maximal steps for nested iteration?",
			     "maxwell_pinvit", 20);
    bloch_steps = askforint("Maximal steps for parameter tracing?",
			    "maxwell_bloch", 40);
  }

  visualization = askforchar("Visualization? (N)one, (F)irst, (A)ll steps?",
			     "maxwell_visual", "nfa", 'n');
  
  gr = (grid2d **) malloc(sizeof(grid2d *) * (L+1));
  e = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  u = (edge2d ***) malloc(sizeof(edge2d **) * (L+1));
  x = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  b = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  d = (edge2d **) malloc(sizeof(edge2d *) * (L+1));
  eprev = 0;
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
    strncpy(permittivityname, "Empty lattice", 40);
    pat = 0;
    break;

  case 'c':
    printf("Permittivity pattern: Circle\n");
    strncpy(permittivityname, "Circle", 40);
    pat = new_circle_epspattern();
    break;

  case 'k':
    printf("Permittivity pattern: Kivshar\n");
    strncpy(permittivityname, "Kivshar", 40);
    pat = new_kivshar_epspattern();
    break;
    
  case 'n':
    printf("Permittivity pattern: Nahid's special pattern #%d\n",
	   nahidpattern);
    snprintf(permittivityname, 40, "Nahid #%d", nahidpattern);
    
    switch(nahidpattern) {
    case 0:
      pat = new_nahidmixed_epspattern();
      break;
      
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
  
  out = fopen(filename, "w");

  fprintf(out,
	  "# Bloch eigenvalues\n"
	  "# ----------------------------------------\n"
	  "# Permittivity pattern: %s\n"
	  "# Coarse grid intervals: %d\n"
	  "# Refinement levels: %d\n"
	  "# Chern vectors: %d\n"
	  "# Extrapolation method: %s\n"
	  "# Stopping criterion: %s\n"
	  "# Eigenvector continuity: %s\n"
	  "# ----------------------------------------\n"
	  "# 1: Bloch step\n"
	  "# 2..%d: Eigenvalues\n"
	  "# %d: Iterations\n"
	  "# %d: Time for this step [s]\n"
	  "# %d: Total time to this step [s]\n"
	  "# %d: Chern partial integral, real part\n"
	  "# %d: Chern partial integral, imaginary part\n"
	  "# ----------------------------------------\n",
	  permittivityname,
	  n, L, chern_vectors,
	  extrapolation == 'n' ? "None" :
	  extrapolation == '1' ? "1st order" : "2nd order",
	  stopping == 'r' ? "Residual" : "Fixed steps",
	  continuity == 'n' ? "None" : continuity == 's' ? "Sign" : "Reorder",
	  eigs+1, eigs+2, eigs+3, eigs+4, eigs+5, eigs+6);
  
  blochcurve(0, &gr[L]->xbloch, &gr[L]->ybloch,
	     &gr[L]->xfactor, &gr[L]->yfactor);
  vertex = true;

  draw2d_grid2d(gr[L], gridfilename);
  
  for(l=L; l-->0; ) {
    gr[l] = coarsen_grid2d(gr[l+1]);

    if(l <= 5) {
      snprintf(filename, 80, "Pictures/grid%02d.png", l);
      draw2d_grid2d(gr[l], filename);
    }
  }
  
  for(l=0; l<=L; l++) {
    e[l] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
    for(i=0; i<eigs; i++)
      e[l][i] = new_edge2d(gr[l]);
    u[l] = (edge2d **) malloc(sizeof(edge2d *) * 3 * eigs);
    for(i=0; i<3*eigs; i++)
      u[l][i] = new_edge2d(gr[l]);
    x[l] = new_edge2d(gr[l]);
    b[l] = new_edge2d(gr[l]);
    d[l] = new_edge2d(gr[l]);
    xg[l] = new_node2d(gr[l]);
    bg[l] = new_node2d(gr[l]);
    dg[l] = new_node2d(gr[l]);
  }

  cdf = new_ncfile(x[L], ieigs, allblochsteps, 64, 64, pat, cdffilename);
  
  eold[0] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
  for(i=0; i<eigs; i++)
    eold[0][i] = new_edge2d(gr[L]);
  eold[1] = (edge2d **) malloc(sizeof(edge2d *) * eigs);
  for(i=0; i<eigs; i++)
    eold[1][i] = new_edge2d(gr[L]);

  chern = (field *) malloc(sizeof(field) * ieigs);
  
  if(visualization == 'a') {
    eprev = (edge2d **) malloc(sizeof(edge2d *) * ieigs);
    for(i=0; i<ieigs; i++)
      eprev[i] = new_edge2d(gr[L]);
  }

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

  Ar = new_matrix(eigs, eigs);
  Mr = new_matrix(eigs, eigs);
  
  Ar2 = new_matrix(2*ieigs, 2*ieigs);
  tau2 = (field *) malloc(sizeof(field) * 2*ieigs);
  lambda2 = (real *) malloc(sizeof(real) * 2*ieigs);
  Ar3 = new_matrix(3*ieigs, 3*ieigs);
  tau3 = (field *) malloc(sizeof(field) * 3*ieigs);
  lambda3 = (real *) malloc(sizeof(real) * 3*ieigs);

  printf("Setting up nodal coarse-grid solver\n");
  An = new_matrix(n*n, n*n);
  densematrix_node2d(true, gr[0], An);
  potrf_matrix(An);

  printf("Setting up edge coarse-grid solver\n");
  Ae = new_matrix(2*n*n, 2*n*n);
  densematrix_edge2d(1.0, 1.0, gr[0], Ae);
  potrf_matrix(Ae);
  
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

#pragma omp parallel private(i,j,k,l,info,t_start,t_stop,t_run,product1,product2,chup,angle,optangle)
  {
    /* ----------------------------------------
     * Nested iteration
     * ---------------------------------------- */

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

#pragma omp single
      {
	printf("----------------------------------------\n"
	       "Level %2d:", l);
	for(i=0; i<ieigs; i++)
	  printf(" %f", lambda[l][i] - 1.0);
	printf("  ");
	for(; i<eigs; i++)
	  printf(" %f", lambda[l][i] - 1.0);
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

	maxnorm = norm[0];
	for(i=1; i<ieigs; i++)
	  if(norm[i] > maxnorm)
	    maxnorm = norm[i];

	fflush(stdout);
      }

      for(k=0; k<pinvit_steps && maxnorm>tolerance; k++) {
	for(i=0; i<eigs; i++) {
	  /* Compute the defect */
	  zero_edge2d(b[l]);
	  addeval_edge2d(1.0, 1.0-lambda[l][i], e[l][i], b[l]);

	  /* Apply the multigrid preconditioner */
	  zero_edge2d(x[l]);
	  for(j=0; j<prec_steps; j++)
	    vcycle_edge2d(l, 2, Ae, b, x, d);
	
	  /* Add update to eigenvector approximation */
	  add_edge2d(-1.0, x[l], e[l][i]);
	  
	  /* Eliminate null-space components */
	  zero_edge2d(d[l]);
	  addeval_edge2d(0.0, 1.0, e[l][i], d[l]);
	  zero_node2d(bg[l]);
	  adjgradient_node2d(1.0, d[l], bg[l]);
	  zero_node2d(xg[l]);
	  for(j=0; j<gradient_steps; j++)
	    vcycle_node2d(l, 2, An, bg, xg, dg);
	  gradient_node2d(-1.0, xg[l], e[l][i]);
	}

#pragma omp barrier
	
	/* Switch old and new approximations */
#pragma omp single
	for(i=0; i<eigs; i++) {
	  aux = u[l][i];
	  u[l][i] = e[l][i];
	  e[l][i] = aux;
	}

	/* Set up Ritz system */
	for(j=0; j<eigs; j++)
	  for(i=j; i<eigs; i++) {
	    Ar->a[i+Ar->ld*j] = energyprod_edge2d(1.0, 1.0, u[l][i], u[l][j]);
	    Mr->a[i+Mr->ld*j] = massprod_edge2d(u[l][i], u[l][j]);
	  }

	/* Solve Ritz system */
#pragma omp single
	{
	  info = sygv_matrix(Ar, Mr, lambda[l]);
	  assert(info == 0);
	}

	/* Construct eigenvector approximations */
	for(j=0; j<eigs; j++) {
	  zero_edge2d(e[l][j]);
	  for(i=0; i<eigs; i++)
	    add_edge2d(Ar->a[i+Ar->ld*j], u[l][i], e[l][j]);
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
	  for(i=0; i<ieigs; i++)
	    printf(" %f", lambda[l][i] - 1.0);
	  printf("  ");
	  for(; i<eigs; i++)
	    printf(" %f", lambda[l][i] - 1.0);
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

	  maxnorm = norm[0];
	  for(i=1; i<ieigs; i++)
	    if(norm[i] > maxnorm)
	      maxnorm = norm[i];
	  
	  fflush(stdout);
	}
      }

      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);
#pragma omp single
      printf("  %.2f seconds for level %d\n", t_run, l);
    }

    write_ncfile(cdf, 0, lambda[L], (const edge2d **) e[L]);

    switch(visualization) {
    default:
      /* Falls through */
    case 'n':
      break;

    case 'a':
      for(i=0; i<ieigs; i++)
	copy_edge2d(e[L][i], eprev[i]);
      /* Falls through */

    case 'f':
      for(i=0; i<ieigs; i++) {
#pragma omp single
	snprintf(filename, 80, "Pictures/eigenvector_step%03d_%d.png", 0, i);
	cairodraw_edge2d(e[L][i], 40, 40, pat, filename);
      }
    }

    /* ----------------------------------------
     * Bloch scan
     * ---------------------------------------- */
    
#pragma omp single
    {
      fprintf(out, "0");
      for(i=0; i<eigs; i++)
	fprintf(out, " %f", lambda[L][i] - 1.0);
      fprintf(out, "\n");
      fflush(out);
    }

    /* Initialize Chern number */
    for(i=0; i<ieigs; i++)
      chern[i] = 0.0;
    sign = 0;

    /* Clear old vectors */
    for(i=0; i<eigs; i++) {
      zero_edge2d(eold[0][i]);
      zero_edge2d(eold[1][i]);
    }
    
    t_accum = 0.0;
    
    for(l=1; l<allblochsteps; l++) {
#pragma omp single
      {
	/* Set new Bloch parameters */
	oldvertex = vertex;
	vertex = blochcurve(l, &gr[L]->xbloch, &gr[L]->ybloch,
			    &gr[L]->xfactor, &gr[L]->yfactor);

	printf("----------------------------------------\n"
	       "Point %3d (%.2f,%.2f)%s\n",
	       l, gr[L]->xbloch, gr[L]->ybloch,
	       (vertex ? " vertex" : ""));
	
	for(i=0; i<L; i++) {
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

      switch(extrapolation) {
      default:
	/* Falls though */
      case 'n':
	/* Update old vectors */
	for(i=0; i<eigs; i++) {
	  copy_edge2d(eold[0][i], eold[1][i]);
	  copy_edge2d(e[L][i], eold[0][i]);
	}
	break;

      case '1':
#pragma omp single
	printf("First-order extrapolation for initial vectors\n");
	
	/* Use old and current vectors */
	for(i=0; i<ieigs; i++) {
	  copy_edge2d(e[L][i], u[L][i]);
	  copy_edge2d(eold[0][i], u[L][i+ieigs]);
	}

	/* Build M-orthonormal basis */
	orthonormalize_edge2d(2*ieigs, u[L], tau2, d[L]);
	
	/* Set up Ritz system */
	for(j=0; j<2*ieigs; j++)
	  for(i=j; i<2*ieigs; i++)
	    Ar2->a[i+Ar2->ld*j] = energyprod_edge2d(1.0, 1.0, u[L][i], u[L][j]);

	/* Solve Ritz system */
#pragma omp single
	{
	  info = syev_matrix(Ar2, lambda2);
	  assert(info == 0);
	}
	  
	/* Construct eigenvector approximations */
	for(j=0; j<eigs; j++) {
	  copy_edge2d(eold[0][j], eold[1][j]);
	  copy_edge2d(e[L][j], eold[0][j]);
	  
	  zero_edge2d(e[L][j]);
	  for(i=0; i<2*ieigs; i++)
	    add_edge2d(Ar2->a[i+Ar2->ld*j], u[L][i], e[L][j]);
	}
	break;

      case '2':
#pragma omp single
	printf("Second-order extrapolation for initial vectors\n");
	
	/* Use very old, old, and current vectors */
	for(i=0; i<ieigs; i++) {
	  copy_edge2d(e[L][i], u[L][i]);
	  copy_edge2d(eold[0][i], u[L][i+ieigs]);
	  copy_edge2d(eold[1][i], u[L][i+2*ieigs]);
	}

	/* Build M-orthonormal basis */
	orthonormalize_edge2d(3*ieigs, u[L], tau3, d[L]);
	
	/* Set up Ritz system */
	for(j=0; j<3*ieigs; j++)
	  for(i=j; i<3*ieigs; i++)
	    Ar3->a[i+Ar3->ld*j] = energyprod_edge2d(1.0, 1.0, u[L][i], u[L][j]);

	/* Solve Ritz system */
#pragma omp single
	{
	  info = syev_matrix(Ar3, lambda3);
	  assert(info == 0);
	}
	  
	/* Construct eigenvector approximations */
	for(j=0; j<eigs; j++) {
	  copy_edge2d(eold[0][j], eold[1][j]);
	  copy_edge2d(e[L][j], eold[0][j]);
	  zero_edge2d(e[L][j]);
	  for(i=0; i<3*ieigs; i++)
	    add_edge2d(Ar3->a[i+Ar3->ld*j], u[L][i], e[L][j]);
	}
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

#pragma omp single
      {
	printf("Ritz values:");
	for(i=0; i<ieigs; i++)
	  printf(" %f", lambda[L][i] - 1.0);
	printf("  ");
	for(; i<eigs; i++)
	  printf(" %f", lambda[L][i] - 1.0);
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

	maxnorm = norm[0];
	for(i=1; i<ieigs; i++)
	  if(norm[i] > maxnorm)
	    maxnorm = norm[i];

	fflush(stdout);
      }

      for(k=0; k<bloch_steps && maxnorm>tolerance; k++) {
	for(i=0; i<eigs; i++) {
	  /* Compute the defect */
	  zero_edge2d(b[L]);
	  addeval_edge2d(1.0, 1.0-lambda[L][i], e[L][i], b[L]);

	  /* Apply the multigrid preconditioner */
	  zero_edge2d(x[L]);
	  for(j=0; j<prec_steps; j++)
	    vcycle_edge2d(L, 2, Ae, b, x, d);
	
	  /* Add update to eigenvector approximation */
	  add_edge2d(-1.0, x[L], e[L][i]);
	  
	  /* Eliminate null-space components */
	  zero_edge2d(d[L]);
	  addeval_edge2d(0.0, 1.0, e[L][i], d[L]);
	  zero_node2d(bg[L]);
	  adjgradient_node2d(1.0, d[L], bg[L]);
	  zero_node2d(xg[L]);
	  for(j=0; j<gradient_steps; j++)
	    vcycle_node2d(L, 2, An, bg, xg, dg);
	  gradient_node2d(-1.0, xg[L], e[L][i]);
	}

#pragma omp barrier
	
	/* Switch old and new approximations */
#pragma omp single
	for(i=0; i<eigs; i++) {
	  aux = u[L][i];
	  u[L][i] = e[L][i];
	  e[L][i] = aux;
	}

	/* Set up Ritz system */
	for(j=0; j<eigs; j++)
	  for(i=j; i<eigs; i++) {
	    Ar->a[i+Ar->ld*j] = energyprod_edge2d(1.0, 1.0, u[L][i], u[L][j]);
	    Mr->a[i+Mr->ld*j] = massprod_edge2d(u[L][i], u[L][j]);
	  }

	/* Solve Ritz system */
#pragma omp single
	{
	  info = sygv_matrix(Ar, Mr, lambda[L]);
	  assert(info == 0);
	}

	/* Construct eigenvector approximations */
	for(j=0; j<eigs; j++) {
	  zero_edge2d(e[L][j]);
	  for(i=0; i<eigs; i++)
	    add_edge2d(Ar->a[i+Ar->ld*j], u[L][i], e[L][j]);
	}
	
	/* Compute defects */
	for(i=0; i<eigs; i++) {
	  zero_edge2d(d[L]);
	  addeval_edge2d(1.0, 1.0-lambda[L][i], e[L][i], d[L]);
	  oldnorm[i] = norm[i];
	  norm[i] = norm2_edge2d(d[L]);
	}

#pragma omp single
	{
	  printf(" Step %2d:", k+1);
	  for(i=0; i<ieigs; i++)
	    printf(" %f", lambda[L][i] - 1.0);
	  printf("  ");
	  for(; i<eigs; i++)
	    printf(" %f", lambda[L][i] - 1.0);
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

	  maxnorm = norm[0];
	  for(i=1; i<ieigs; i++)
	    if(norm[i] > maxnorm)
	      maxnorm = norm[i];

	fflush(stdout);
	}
      }

      t_stop = omp_get_wtime();
      t_run = reduce_max_real(t_stop - t_start);

      /* Adjust signs */
      switch(continuity) {
      default:
	/* Falls through */
      case 'n':
	break;

      case 'r':
	if(oldvertex) {
	  for(i=0; i<ieigs; i++) {
	    /* Use the old eigenvector as a guess for the next eigenvector */
	    copy_edge2d(eold[0][i], x[L]);

	    /* Compute the cosine of the angle between the eigenvector
	     * and the guess */
	    optangle = cabs(massprod_edge2d(e[L][i], x[L]));
	    k = i;

	    /* Find the eigenvector closest to the guess */
	    for(j=i+1; j<ieigs; j++) {
	      angle = cabs(massprod_edge2d(e[L][j], x[L]));
	      if(angle > optangle) {
		k = j;
		optangle = angle;
	      }
	    }
	    
	    /* Swap the eigenvectors if necessary */
	    if(k != i) {
	      swap_edge2d(e[L][i], e[L][k]);
	      
#pragma omp single
	      {
		swap = lambda[L][i];
		lambda[L][i] = lambda[L][k];
		lambda[L][k] = swap;

		sign++;
	      }
	    }
	  }
	}
	else {
	  for(i=0; i<ieigs; i++) {
	    /* Extrapolate a guess for the next eigenvector */
	    copy_edge2d(eold[0][i], x[L]);
	    scale_edge2d(2.0, x[L]);
	    add_edge2d(-1.0, eold[1][i], x[L]);

	    /* Make it a unit vector */
	    angle = sqrt(creal(massprod_edge2d(x[L], x[L])));
	    scale_edge2d(1.0/angle, x[L]);

	    /* Compute the cosine of the angle between the eigenvector
	     * and the guess */
	    optangle = cabs(massprod_edge2d(e[L][i], x[L]));
	    k = i;

	    /* Find the eigenvector closest to the guess */
	    for(j=i+1; j<ieigs; j++) {
	      angle = cabs(massprod_edge2d(e[L][j], x[L]));
	      if(angle > optangle) {
		k = j;
		optangle = angle;
	      }
	    }
	    
	    /* Swap the eigenvectors if necessary */
	    if(k != i) {
	      swap_edge2d(e[L][i], e[L][k]);
	      
#pragma omp single
	      {
		swap = lambda[L][i];
		lambda[L][i] = lambda[L][k];
		lambda[L][k] = swap;

		sign++;
	      }
	    }
	  }
	}
	/* Falls through */
	
      case 's':
	if(oldvertex) {
	  /* Only one previous eigenvector, so we adjust the sign
	   * to match it */
	  for(i=0; i<eigs; i++) {
	    product1 = massprod_edge2d(e[L][i], eold[0][i]);
	    product1 /= ABS(product1);
	    scale_edge2d(product1, e[L][i]);
	  }
	}
	else {
	  for(i=0; i<eigs; i++) {
	    /* Extrapolate using two previous eigenvectors, adjust
	     * the sign to match the predicted eigenvector */
	    copy_edge2d(eold[0][i], x[L]);
	    scale_edge2d(2.0, x[L]);
	    add_edge2d(-1.0, eold[1][i], x[L]);
	    product1 = massprod_edge2d(e[L][i], x[L]);
	    product1 /= ABS(product1);
	    scale_edge2d(product1, e[L][i]);
	  }
	}
      }
	
      /* DEBUGGING: Test angle between old and new eigenvectors
#pragma omp single
      printf(" Cosines:");
      for(i=0; i<eigs; i++) {
	product1 = massprod_edge2d(eold[0][i], e[L][i]);
#pragma omp single
	printf(" (%.2f,%.2f)", creal(product1), cimag(product1));
      }
#pragma omp single
      printf("\n");
      */
      
      /* Update Chern sums */
      if(oldvertex) {
	/* We have just (re)started and eold[1] is not valid,
	 * so we can use only a first-order approximation */

	for(i=0; i<chern_vectors; i++) {
	  /* Compute forward difference */
	  copy_edge2d(e[L][i], x[L]);
	  add_edge2d(-1.0, eold[0][i], x[L]);
	  chup = 0.5 * I * massprod_edge2d(eold[0][i], x[L]);

#pragma omp single
	  chern[i] += chup;
	}
      }
      else {
	/* eold[0] and eold[1] are valid, so we can approximate
	 * the derivative for eold[0] by a second-order approximation */

	for(i=0; i<chern_vectors; i++) {
	  /* Compute central difference */
	  copy_edge2d(e[L][i], x[L]);
	  add_edge2d(-1.0, eold[1][i], x[L]);
	  chup = 0.5 * I * massprod_edge2d(eold[0][i], x[L]);

	  if(vertex) {
	    /* We will restart next, so we have to handle the
	     * next point by first-order approximation */
	    
	    /* Compute backward difference */
	    copy_edge2d(e[L][i], x[L]);
	    add_edge2d(-1.0, eold[0][i], x[L]);
	    chup += 0.5 * I * massprod_edge2d(e[L][i], x[L]);
	  }

#pragma omp single
	  chern[i] += chup;
	}
      }
      
#pragma omp single
      {
	printf("   Chern:");
	for(i=0; i<chern_vectors; i++)
	  printf(" %.2f", creal(chern[i]));
	printf("  %d\n", sign);

	t_accum += t_run;
      
	fprintf(out, "%d", l);
	for(i=0; i<eigs; i++)
	  fprintf(out, " %f", lambda[L][i] - 1.0);
	fprintf(out, "  %d  %.1f %.1f ",
		k, t_run, t_accum);
	for(i=0; i<chern_vectors; i++)
	  fprintf(out, " %f %f", creal(chern[i]), cimag(chern[i]));
	fprintf(out, "  %d\n", sign);
	fflush(out);
      }

      switch(visualization) {
      default:
	/* Falls through */
      case 'n':
	/* Falls through */
      case 'f':
	break;

      case 'a':
	for(i=0; i<ieigs; i++) {
	  alpha = dotprod_edge2d(eprev[i], e[L][i]);
	  alpha /= ABS(alpha);
	  scale_edge2d(CONJ(alpha), e[L][i]);
	  copy_edge2d(e[L][i], eprev[i]);
	}
	
	for(i=0; i<ieigs; i++) {
#pragma omp single
	  snprintf(filename, 80, "Pictures/eigenvector_step%03d_%d.png", l, i);
	  cairodraw_edge2d(e[L][i], 40, 40, pat, filename);
	}
      }
    
      write_ncfile(cdf, l, lambda[L], (const edge2d **) e[L]);

#pragma omp single
      printf("  %.2f seconds (%.2f seconds per step) for point %d\n",
	     t_run, t_run / k, l);
    }
  }

  del_ncfile(cdf);
  
  return 0;
}

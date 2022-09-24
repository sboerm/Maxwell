
#include "node2d.h"

#include "basic.h"
#include "linalg.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

/* DEBUGGING */
#include <stdio.h>

node2d *
new_node2d(const grid2d *gr)
{
  node2d *x;

  x = (node2d *) malloc(sizeof(node2d));
  x->gr = gr;
  x->v = (field *) malloc(sizeof(field) * gr->nx * gr->ny);

  return x;
}

void
del_node2d(node2d *x)
{
  free(x->v);
  x->v = 0;

  x->gr = 0;

  free(x);
}

void
zero_node2d(node2d *x)
{
  field *xv = x->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  /* Clear coefficients */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xv[i+j*inc] = 0.0;
}

void
copy_node2d(const node2d *x, node2d *y)
{
  const field *xv = x->v;
  field *yv = y->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  assert(x->gr == y->gr);
  
  /* Copy nodes */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yv[i+j*inc] = xv[i+j*inc];
}

void
random_node2d(node2d *x)
{
  field *xv = x->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  /* Set random coefficients for nodes */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      xv[i+j*inc] = 2.0 * rand() / RAND_MAX - 1.0;
#ifdef FIELD_COMPLEX
      xv[i+j*inc] += I * (2.0 * rand() / RAND_MAX - 1.0);
#endif
    }
}

void
interpolate_node2d(node2d *x,
		   field (*func)(const real *x, void *data),
		   void *data)
{
  field *xv = x->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  real xp[2];
  int i, j;

  /* Interpolate on nodes */
#pragma omp for
  for(j=0; j<ny; j++) {
    xp[1] = j * hy;
    for(i=0; i<nx; i++) {
      xp[0] = i * hx;
      xv[i+j*inc] = func(xp, data);
    }
  }
}

void
l2functional_node2d(node2d *x, node2d *buf,
		    field (*func)(const real *x, void *data),
		    void *data)
{
  field *xv = x->v;
  field *bufv = buf->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  const real xq0 = 0.5 * (1.0 - sqrt(1.0/3.0));
  const real xq1 = 0.5 * (1.0 + sqrt(1.0/3.0));
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  real xp[2];
  int i, j;

  assert(buf->gr == x->gr);

  zero_node2d(x);
  
  /* Evaluate function in the first quadrature point */
#pragma omp for
  for(j=0; j<ny; j++) {
    xp[1] = (j + xq0) * hy;
    for(i=0; i<nx; i++) {
      xp[0] = (i + xq0) * hx;

      bufv[i+j*inc] = func(xp, data);
    }
  }

  /* Add contribution to all basis functions */
#pragma omp single
  {
    xv[0] += (xq1 * xq1 * bufv[0] +
	      xq0 * xq1 * CONJ(xfactor) * bufv[nx-1] +
	      xq1 * xq0 * CONJ(yfactor) * bufv[0+(ny-1)*inc] +
	      xq0 * xq0 * CONJ(xfactor) * CONJ(yfactor) * bufv[(nx-1)+(ny-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i] += (xq1 * xq1 * bufv[i] +
		xq0 * xq1 * bufv[i-1] +
		xq1 * xq0 * CONJ(yfactor) * bufv[i+(ny-1)*inc] +
		xq0 * xq0 * CONJ(yfactor) * bufv[(i-1)+(ny-1)*inc]);
  }
  
#pragma omp for
  for(j=1; j<ny; j++) {
    xv[j*inc] += (xq1 * xq1 * bufv[0+j*inc] +
		  xq0 * xq1 * CONJ(xfactor) * bufv[(nx-1)+j*inc] +
		  xq1 * xq0 * bufv[0+(j-1)*inc] +
		  xq0 * xq0 * CONJ(xfactor) * bufv[(nx-1)+(j-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i+j*inc] += (xq1 * xq1 * bufv[i+j*inc] +
		      xq0 * xq1 * bufv[(i-1)+j*inc] +
		      xq1 * xq0 * bufv[i+(j-1)*inc] +
		      xq0 * xq0 * bufv[(i-1)+(j-1)*inc]);
  }
  
  /* Evaluate function in the second quadrature point */
#pragma omp for
  for(j=0; j<ny; j++) {
    xp[1] = (j + xq0) * hy;
    for(i=0; i<nx; i++) {
      xp[0] = (i + xq1) * hx;

      bufv[i+j*inc] = func(xp, data);
    }
  }

  /* Add contribution to all basis functions */
#pragma omp single
  {
    xv[0] += (xq0 * xq1 * bufv[0] +
	      xq1 * xq1 * CONJ(xfactor) * bufv[nx-1] +
	      xq0 * xq0 * CONJ(yfactor) * bufv[0+(ny-1)*inc] +
	      xq1 * xq0 * CONJ(xfactor) * CONJ(yfactor) * bufv[(nx-1)+(ny-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i] += (xq0 * xq1 * bufv[i] +
		xq1 * xq1 * bufv[i-1] +
		xq0 * xq0 * CONJ(yfactor) * bufv[i+(ny-1)*inc] +
		xq1 * xq0 * CONJ(yfactor) * bufv[(i-1)+(ny-1)*inc]);
  }
  
#pragma omp for
  for(j=1; j<ny; j++) {
    xv[j*inc] += (xq0 * xq1 * bufv[0+j*inc] +
		  xq1 * xq1 * CONJ(xfactor) * bufv[(nx-1)+j*inc] +
		  xq0 * xq0 * bufv[0+(j-1)*inc] +
		  xq1 * xq0 * CONJ(xfactor) * bufv[(nx-1)+(j-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i+j*inc] += (xq0 * xq1 * bufv[i+j*inc] +
		      xq1 * xq1 * bufv[(i-1)+j*inc] +
		      xq0 * xq0 * bufv[i+(j-1)*inc] +
		      xq1 * xq0 * bufv[(i-1)+(j-1)*inc]);
  }
  
  /* Evaluate function in the third quadrature point */
#pragma omp for
  for(j=0; j<ny; j++) {
    xp[1] = (j + xq1) * hy;
    for(i=0; i<nx; i++) {
      xp[0] = (i + xq0) * hx;

      bufv[i+j*inc] = func(xp, data);
    }
  }

  /* Add contribution to all basis functions */
#pragma omp single
  {
    xv[0] += (xq1 * xq0 * bufv[0] +
	      xq0 * xq0 * CONJ(xfactor) * bufv[nx-1] +
	      xq1 * xq1 * CONJ(yfactor) * bufv[0+(ny-1)*inc] +
	      xq0 * xq1 * CONJ(xfactor) * CONJ(yfactor) * bufv[(nx-1)+(ny-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i] += (xq1 * xq0 * bufv[i] +
		xq0 * xq0 * bufv[i-1] +
		xq1 * xq1 * CONJ(yfactor) * bufv[i+(ny-1)*inc] +
		xq0 * xq1 * CONJ(yfactor) * bufv[(i-1)+(ny-1)*inc]);
  }
  
#pragma omp for
  for(j=1; j<ny; j++) {
    xv[j*inc] += (xq1 * xq0 * bufv[0+j*inc] +
		  xq0 * xq0 * CONJ(xfactor) * bufv[(nx-1)+j*inc] +
		  xq1 * xq1 * bufv[0+(j-1)*inc] +
		  xq0 * xq1 * CONJ(xfactor) * bufv[(nx-1)+(j-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i+j*inc] += (xq1 * xq0 * bufv[i+j*inc] +
		      xq0 * xq0 * bufv[(i-1)+j*inc] +
		      xq1 * xq1 * bufv[i+(j-1)*inc] +
		      xq0 * xq1 * bufv[(i-1)+(j-1)*inc]);
  }
  
  /* Evaluate function in the fourth quadrature point */
#pragma omp for
  for(j=0; j<ny; j++) {
    xp[1] = (j + xq1) * hy;
    for(i=0; i<nx; i++) {
      xp[0] = (i + xq1) * hx;

      bufv[i+j*inc] = func(xp, data);
    }
  }

  /* Add contribution to all basis functions */
#pragma omp single
  {
    xv[0] += (xq0 * xq0 * bufv[0] +
	      xq1 * xq0 * CONJ(xfactor) * bufv[nx-1] +
	      xq0 * xq1 * CONJ(yfactor) * bufv[0+(ny-1)*inc] +
	      xq1 * xq1 * CONJ(xfactor) * CONJ(yfactor) * bufv[(nx-1)+(ny-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i] += (xq0 * xq0 * bufv[i] +
		xq1 * xq0 * bufv[i-1] +
		xq0 * xq1 * CONJ(yfactor) * bufv[i+(ny-1)*inc] +
		xq1 * xq1 * CONJ(yfactor) * bufv[(i-1)+(ny-1)*inc]);
  }
  
#pragma omp for
  for(j=1; j<ny; j++) {
    xv[j*inc] += (xq0 * xq0 * bufv[0+j*inc] +
		  xq1 * xq0 * CONJ(xfactor) * bufv[(nx-1)+j*inc] +
		  xq0 * xq1 * bufv[0+(j-1)*inc] +
		  xq1 * xq1 * CONJ(xfactor) * bufv[(nx-1)+(j-1)*inc]);

    for(i=1; i<nx; i++)
      xv[i+j*inc] += (xq0 * xq0 * bufv[i+j*inc] +
		      xq1 * xq0 * bufv[(i-1)+j*inc] +
		      xq0 * xq1 * bufv[i+(j-1)*inc] +
		      xq1 * xq1 * bufv[(i-1)+(j-1)*inc]);
  }
  
  /* Multiply by quadrature weight */
  scale_node2d(0.25 * hx * hy, x);
}

real
l2norm_node2d(const node2d *x,
	      field (*func)(const real *x, void *data),
	      void *data)
{
  const field *xv = x->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  const real xq0 = 0.5 * (1.0 - sqrt(1.0/3.0));
  const real xq1 = 0.5 * (1.0 + sqrt(1.0/3.0));
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  field val[2][2], vl;
  real xp[2];
  real sum;
  int i, j;

  sum = 0.0;

#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      val[0][0] = xv[i+j*inc];
      val[1][0] = (i < nx-1 ? xv[(i+1)+j*inc] : xfactor * xv[0+j*inc]);
      val[0][1] = (j < ny-1 ? xv[i+(j+1)*inc] : yfactor * xv[i]);
      val[1][1] = (j < ny-1 ?
		   (i < nx-1 ? xv[(i+1)+(j+1)*inc] :
		    xfactor * xv[0+(j+1)*inc]) :
		   (i < nx-1 ? yfactor * xv[i+1] :
		    xfactor * yfactor * xv[0]));
      
      /* First quadrature point */
      xp[0] = (i + xq0) * hx;
      xp[1] = (j + xq0) * hy;
      vl = (xq1 * xq1 * val[0][0] +
	    xq0 * xq1 * val[1][0] +
	    xq1 * xq0 * val[0][1] +
	    xq0 * xq0 * val[1][1]);
      if(func)
	vl -= func(xp, data);
      sum += ABS2(vl);

      /* Second quadrature point */
      xp[0] = (i + xq1) * hx;
      vl = (xq0 * xq1 * val[0][0] +
	    xq1 * xq1 * val[1][0] +
	    xq0 * xq0 * val[0][1] +
	    xq1 * xq0 * val[1][1]);
      if(func)
	vl -= func(xp, data);
      sum += ABS2(vl);

      /* Third quadrature point */
      xp[1] = (j + xq1) * hy;
      vl = (xq0 * xq0 * val[0][0] +
	    xq1 * xq0 * val[1][0] +
	    xq0 * xq1 * val[0][1] +
	    xq1 * xq1 * val[1][1]);
      if(func)
	vl -= func(xp, data);
      sum += ABS2(vl);

      /* Fourth quadrature point */
      xp[0] = (i + xq0) * hx;
      vl = (xq1 * xq0 * val[0][0] +
	    xq0 * xq0 * val[1][0] +
	    xq1 * xq1 * val[0][1] +
	    xq0 * xq1 * val[1][1]);
      if(func)
	vl -= func(xp, data);
      sum += ABS2(vl);
    }

  sum *= 0.25 * hx * hy;

  return sqrt(reduce_sum_real(sum));
}

void
scale_node2d(field alpha, node2d *x)
{
  field *xv = x->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  /* Scale nodal values */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xv[i+j*inc] *= alpha;
}

void
add_node2d(field alpha, const node2d *x, node2d *y)
{
  const field *xv = x->v;
  field *yv = y->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  assert(x->gr == y->gr);
  
  /* Add nodal values */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yv[i+j*inc] += alpha * xv[i+j*inc];
}

real
norm2_node2d(const node2d *x)
{
  const field *xv = x->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  real sum;
  int i, j;

  sum = 0.0;
  
  /* Add squares of x nodes */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += ABS2(xv[i+j*inc]);

  return sqrt(reduce_sum_real(sum));
}

real
normmax_node2d(const node2d *x)
{
  const field *xv = x->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  real amax, aval;
  int i, j;

  amax = 0.0;
  
  /* Maximum of x nodes */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      aval = ABS(xv[i+j*inc]);
      if(aval > amax)
	amax = aval;
    }

  return reduce_max_real(amax);
}

field
dotprod_node2d(const node2d *x, const node2d *y)
{
  const field *xv = x->v;
  const field *yv = y->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  field sum;
  int i, j;

  assert(x->gr == y->gr);

  sum = 0.0;
  
  /* Add products */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += CONJ(xv[i+j*inc]) * yv[i+j*inc];

  return reduce_sum_field(sum);
}

void
center_node2d(node2d *x)
{
  field *xv = x->v;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  field sum;
  int i, j;

  sum = 0.0;
  
  /* Add coefficients */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += xv[i+j*inc];

  sum = reduce_sum_field(sum) / (nx * ny);

  /* Subtract average */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xv[i+j*inc] -= sum;
}

void
addeval_node2d(field alpha,
	       const node2d *x, node2d *y)
{
  const field *xv = x->v;
  field *yv = y->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ac = 4.0 * alpha * (hx / hy + hy / hx) / 3.0;
  field ax = alpha * (hx / hy - 2.0 * hy / hx) / 3.0;
  field ay = alpha * (hy / hx - 2.0 * hx / hy) / 3.0;
  field ad = -alpha * (hx / hy + hy / hx) / 6.0;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  assert(x->gr == y->gr);

  /* Evaluate alpha*A for the first row */
#pragma omp single
  {
    yv[0] += (ac * xv[0]
	      + ax * (CONJ(xfactor) * xv[nx-1] + xv[1])
	      + ay * (CONJ(yfactor) * xv[0+(ny-1)*inc] + xv[0+inc])
	      + ad * (CONJ(xfactor) * CONJ(yfactor) * xv[(nx-1)+(ny-1)*inc]
		      + CONJ(yfactor) * xv[1+(ny-1)*inc]
		      + CONJ(xfactor) * xv[(nx-1)+inc]
		      + xv[1+inc]));

    for(i=1; i<nx-1; i++)
      yv[i] += (ac * xv[i]
		+ ax * (xv[i-1] + xv[i+1])
		+ ay * (CONJ(yfactor) * xv[i+(ny-1)*inc] + xv[i+inc])
		+ ad * (CONJ(yfactor) * xv[(i-1)+(ny-1)*inc]
			+ CONJ(yfactor) * xv[(i+1)+(ny-1)*inc]
			+ xv[(i-1)+inc]
			+ xv[(i+1)+inc]));

    yv[nx-1] += (ac * xv[nx-1]
		 + ax * (xv[nx-2] + xfactor * xv[0])
		 + ay * (CONJ(yfactor) * xv[(nx-1)+(ny-1)*inc]
			 + xv[(nx-1)+inc])
		 + ad * (CONJ(yfactor) * xv[(nx-2)+(ny-1)*inc]
			 + xfactor * CONJ(yfactor) * xv[0+(ny-1)*inc]
			 + xv[(nx-2)+inc]
			 + xfactor * xv[0+inc]));
  }

  /* Evaluate alpha*A for intermediate rows */
#pragma omp for
  for(j=1; j<ny-1; j++) {
    yv[0+j*inc] += (ac * xv[0+j*inc]
		    + ax * (CONJ(xfactor) * xv[(nx-1)+j*inc] + xv[1+j*inc])
		    + ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		    + ad * (CONJ(xfactor) * xv[(nx-1)+(j-1)*inc]
			    + xv[1+(j-1)*inc]
			    + CONJ(xfactor) * xv[(nx-1)+(j+1)*inc]
			    + xv[1+(j+1)*inc]));

    for(i=1; i<nx-1; i++)
      yv[i+j*inc] += (ac * xv[i+j*inc]
		      + ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		      + ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		      + ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			      + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc]));
    
    yv[(nx-1)+j*inc] += (ac * xv[(nx-1)+j*inc]
			 + ax * (xv[(nx-2)+j*inc]
				 + xfactor * xv[0+j*inc])
			 + ay * (xv[(nx-1)+(j-1)*inc]
				 + xv[(nx-1)+(j+1)*inc])
			 + ad * (xv[(nx-2)+(j-1)*inc]
				 + xfactor * xv[0+(j-1)*inc]
				 + xv[(nx-2)+(j+1)*inc]
				 + xfactor * xv[0+(j+1)*inc]));
  }

  /* Evaluate alpha*A for the last row */
#pragma omp single
  {
    yv[0+(ny-1)*inc] += (ac * xv[(ny-1)*inc]
			 + ax * (CONJ(xfactor) * xv[(nx-1)+(ny-1)*inc]
				 + xv[1+(ny-1)*inc])
			 + ay * (xv[0+(ny-2)*inc] + yfactor * xv[0])
			 + ad * (CONJ(xfactor) * xv[(nx-1)+(ny-2)*inc]
				 + xv[1+(ny-2)*inc]
				 + CONJ(xfactor) * yfactor * xv[nx-1]
				 + yfactor * xv[1]));
    
    for(i=1; i<nx-1; i++)
      yv[i+(ny-1)*inc] += (ac * xv[i+(ny-1)*inc]
			   + ax * (xv[(i-1)+(ny-1)*inc] + xv[(i+1)+(ny-1)*inc])
			   + ay * (xv[i+(ny-2)*inc] + yfactor * xv[i])
			   + ad * (xv[(i-1)+(ny-2)*inc]
				   + xv[(i+1)+(ny-2)*inc]
				   + yfactor * xv[i-1]
				   + yfactor * xv[i+1]));

    yv[(nx-1)+(ny-1)*inc] += (ac * xv[(nx-1)+(ny-1)*inc]
			      + ax * (xv[(nx-2)+(ny-1)*inc]
				      + xfactor * xv[0+(ny-1)*inc])
			      + ay * (xv[(nx-1)+(ny-2)*inc]
				      + yfactor * xv[nx-1])
			      + ad * (xv[(nx-2)+(ny-2)*inc]
				      + xfactor * xv[0+(ny-2)*inc]
				      + yfactor * xv[nx-2]
				      + xfactor * yfactor * xv[0]));
  }
}

void
gsforward_node2d(field alpha,
		 const node2d *b, node2d *x)
{
  const field *bv = b->v;
  field *xv = x->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(x->gr->xfactor);
  field iyfactor = CONJ(x->gr->yfactor);
  field ac = 4.0 * alpha * (hx / hy + hy / hx) / 3.0;
  field ax = alpha * (hx / hy - 2.0 * hy / hx) / 3.0;
  field ay = alpha * (hy / hx - 2.0 * hx / hy) / 3.0;
  field ad = -alpha * (hx / hy + hy / hx) / 6.0;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  assert(x->gr == b->gr);

#pragma omp single
  {
    xv[0] = (bv[0]
	     - ax * (ixfactor * xv[nx-1] + xv[1])
	     - ay * (iyfactor * xv[0+(ny-1)*inc] + xv[0+inc])
	     - ad * (ixfactor * iyfactor * xv[(nx-1)+(ny-1)*inc]
		     + iyfactor * xv[1+(ny-1)*inc]
		     + ixfactor * xv[(nx-1)+inc]
		     + xv[1+inc])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i] = (bv[i]
	       - ax * (xv[i-1] + xv[i+1])
	       - ay * (iyfactor * xv[i+(ny-1)*inc] + xv[i+inc])
	       - ad * (iyfactor * xv[(i-1)+(ny-1)*inc]
		       + iyfactor * xv[(i+1)+(ny-1)*inc]
		       + xv[(i-1)+inc]
		       + xv[(i+1)+inc])) / ac;

    xv[nx-1] = (bv[nx-1]
		- ax * (xv[nx-2] + xfactor * xv[0])
		- ay * (iyfactor * xv[(nx-1)+(ny-1)*inc]
			+ xv[(nx-1)+inc])
		- ad * (iyfactor * xv[(nx-2)+(ny-1)*inc]
			+ xfactor * iyfactor * xv[0+(ny-1)*inc]
			+ xv[(nx-2)+inc]
			+ xfactor * xv[0+inc])) / ac;
  }
  
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc]
				+ xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc]
				+ xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
  }

#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc]
				+ xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc]
				+ xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
  }

#pragma omp single
  {
    xv[0+(ny-1)*inc] = (bv[0+(ny-1)*inc]
			- ax * (ixfactor * xv[(nx-1)+(ny-1)*inc]
				+ xv[1+(ny-1)*inc])
		   - ay * (xv[0+(ny-2)*inc] + yfactor * xv[0])
		   - ad * (ixfactor * xv[(nx-1)+(ny-2)*inc]
			   + xv[1+(ny-2)*inc]
			   + ixfactor * yfactor * xv[nx-1]
			   + yfactor * xv[1])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i+(ny-1)*inc] = (bv[i+(ny-1)*inc]
			  - ax * (xv[(i-1)+(ny-1)*inc] + xv[(i+1)+(ny-1)*inc])
			  - ay * (xv[i+(ny-2)*inc] + yfactor * xv[i])
			  - ad * (xv[(i-1)+(ny-2)*inc]
				  + xv[(i+1)+(ny-2)*inc]
				  + yfactor * xv[i-1]
				  + yfactor * xv[i+1])) / ac;

    xv[(nx-1)+(ny-1)*inc] = (bv[(nx-1)+(ny-1)*inc]
			- ax * (xv[(nx-2)+(ny-1)*inc]
				+ xfactor * xv[0+(ny-1)*inc])
			- ay * (xv[(nx-1)+(ny-2)*inc]
				+ yfactor * xv[nx-1])
			- ad * (xv[(nx-2)+(ny-2)*inc]
				+ xfactor * xv[0+(ny-2)*inc]
				+ yfactor * xv[nx-2]
				+ xfactor * yfactor * xv[0])) / ac;
  }
}

void
gsbackward_node2d(field alpha,
		 const node2d *b, node2d *x)
{
  const field *bv = b->v;
  field *xv = x->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(x->gr->xfactor);
  field iyfactor = CONJ(x->gr->yfactor);
  field ac = 4.0 * alpha * (hx / hy + hy / hx) / 3.0;
  field ax = alpha * (hx / hy - 2.0 * hy / hx) / 3.0;
  field ay = alpha * (hy / hx - 2.0 * hx / hy) / 3.0;
  field ad = -alpha * (hx / hy + hy / hx) / 6.0;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  assert(x->gr == b->gr);

#pragma omp single
  {
    xv[(nx-1)+(ny-1)*inc] = (bv[(nx-1)+(ny-1)*inc]
			     - ax * (xv[(nx-2)+(ny-1)*inc]
				     + xfactor * xv[0+(ny-1)*inc])
			     - ay * (xv[(nx-1)+(ny-2)*inc]
				     + yfactor * xv[nx-1])
			     - ad * (xv[(nx-2)+(ny-2)*inc]
				     + xfactor * xv[0+(ny-2)*inc]
				     + yfactor * xv[nx-2]
				     + xfactor * yfactor * xv[0])) / ac;

    for(i=nx-2; i>0; i--)
      xv[i+(ny-1)*inc] = (bv[i+(ny-1)*inc]
			  - ax * (xv[(i-1)+(ny-1)*inc] + xv[(i+1)+(ny-1)*inc])
			  - ay * (xv[i+(ny-2)*inc] + yfactor * xv[i])
			  - ad * (xv[(i-1)+(ny-2)*inc]
				  + xv[(i+1)+(ny-2)*inc]
				  + yfactor * xv[i-1]
				  + yfactor * xv[i+1])) / ac;

    xv[0+(ny-1)*inc] = (bv[0+(ny-1)*inc]
			- ax * (ixfactor * xv[(nx-1)+(ny-1)*inc]
				+ xv[1+(ny-1)*inc])
			- ay * (xv[0+(ny-2)*inc] + yfactor * xv[0])
			- ad * (ixfactor * xv[(nx-1)+(ny-2)*inc]
				+ xv[1+(ny-2)*inc]
				+ ixfactor * yfactor * xv[nx-1]
				+ yfactor * xv[1])) / ac;
  }

#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc]
				+ xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc]
				+ xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
    
    for(i=nx-2; i>0; i--)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
  }

#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
    
    for(i=nx-2; i>0; i--)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
  }

#pragma omp single
  {
    xv[nx-1] = (bv[nx-1]
		- ax * (xv[nx-2] + xfactor * xv[0])
		- ay * (iyfactor * xv[(nx-1)+(ny-1)*inc]
			+ xv[(nx-1)+inc])
		- ad * (iyfactor * xv[(nx-2)+(ny-1)*inc]
			+ xfactor * iyfactor * xv[0+(ny-1)*inc]
			+ xv[(nx-2)+inc]
			+ xfactor * xv[0+inc])) / ac;
    
    for(i=nx-2; i>0; i--)
      xv[i] = (bv[i]
	       - ax * (xv[i-1] + xv[i+1])
	       - ay * (iyfactor * xv[i+(ny-1)*inc] + xv[i+inc])
	       - ad * (iyfactor * xv[(i-1)+(ny-1)*inc]
		       + iyfactor * xv[(i+1)+(ny-1)*inc]
		       + xv[(i-1)+inc]
		       + xv[(i+1)+inc])) / ac;

    xv[0] = (bv[0]
	     - ax * (ixfactor * xv[nx-1] + xv[1])
	     - ay * (iyfactor * xv[0+(ny-1)*inc] + xv[0+inc])
	     - ad * (ixfactor * iyfactor * xv[(nx-1)+(ny-1)*inc]
		     + iyfactor * xv[1+(ny-1)*inc]
		     + ixfactor * xv[(nx-1)+inc]
		     + xv[1+inc])) / ac;
  }
}

void
gssymm_node2d(field alpha,
	      const node2d *b, node2d *x)
{
  const field *bv = b->v;
  field *xv = x->v;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(x->gr->xfactor);
  field iyfactor = CONJ(x->gr->yfactor);
  field ac = 4.0 * alpha * (hx / hy + hy / hx) / 3.0;
  field ax = alpha * (hx / hy - 2.0 * hy / hx) / 3.0;
  field ay = alpha * (hy / hx - 2.0 * hx / hy) / 3.0;
  field ad = -alpha * (hx / hy + hy / hx) / 6.0;
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int i, j;

  assert(x->gr == b->gr);

#pragma omp single
  {
    xv[0] = (bv[0]
	     - ax * (ixfactor * xv[nx-1] + xv[1])
	     - ay * (iyfactor * xv[0+(ny-1)*inc] + xv[0+inc])
	     - ad * (ixfactor * iyfactor * xv[(nx-1)+(ny-1)*inc]
		     + iyfactor * xv[1+(ny-1)*inc]
		     + ixfactor * xv[(nx-1)+inc]
		     + xv[1+inc])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i] = (bv[i]
	       - ax * (xv[i-1] + xv[i+1])
	       - ay * (iyfactor * xv[i+(ny-1)*inc] + xv[i+inc])
	       - ad * (iyfactor * xv[(i-1)+(ny-1)*inc]
		       + iyfactor * xv[(i+1)+(ny-1)*inc]
		       + xv[(i-1)+inc] + xv[(i+1)+inc])) / ac;

    xv[nx-1] = (bv[nx-1]
		- ax * (xv[nx-2] + xfactor * xv[0])
		- ay * (iyfactor * xv[(nx-1)+(ny-1)*inc] + xv[(nx-1)+inc])
		- ad * (iyfactor * xv[(nx-2)+(ny-1)*inc]
			+ xfactor * iyfactor * xv[0+(ny-1)*inc]
			+ xv[(nx-2)+inc]
			+ xfactor * xv[0+inc])) / ac;
  }
  
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
		   - ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
  }

#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
		   - ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
  }

#pragma omp single
  {
    j = ny-1;
    
    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + yfactor * xv[0])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * yfactor * xv[nx-1]
			   + yfactor * xv[1])) / ac;
    
    for(i=1; i<nx-1; i++)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + yfactor * xv[i])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + yfactor * xv[i-1] + yfactor * xv[i+1])) / ac;

    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + yfactor * xv[nx-1])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ yfactor * xv[nx-2]
				+ xfactor * yfactor * xv[0])) / ac;

    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + yfactor * xv[nx-1])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ yfactor * xv[nx-2]
				+ xfactor * yfactor * xv[0])) / ac;

    for(i=nx-2; i>0; i--)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + yfactor * xv[i])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + yfactor * xv[i-1] + yfactor * xv[i+1])) / ac;

    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + yfactor * xv[0])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * yfactor * xv[nx-1]
			   + yfactor * xv[1])) / ac;
  }

#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
    
    for(i=nx-2; i>0; i--)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
  }

#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    xv[(nx-1)+j*inc] = (bv[(nx-1)+j*inc]
			- ax * (xv[(nx-2)+j*inc] + xfactor * xv[0+j*inc])
			- ay * (xv[(nx-1)+(j-1)*inc] + xv[(nx-1)+(j+1)*inc])
			- ad * (xv[(nx-2)+(j-1)*inc]
				+ xfactor * xv[0+(j-1)*inc]
				+ xv[(nx-2)+(j+1)*inc]
				+ xfactor * xv[0+(j+1)*inc])) / ac;
    
    for(i=nx-2; i>0; i--)
      xv[i+j*inc] = (bv[i+j*inc]
		     - ax * (xv[(i-1)+j*inc] + xv[(i+1)+j*inc])
		     - ay * (xv[i+(j-1)*inc] + xv[i+(j+1)*inc])
		     - ad * (xv[(i-1)+(j-1)*inc] + xv[(i+1)+(j-1)*inc]
			     + xv[(i-1)+(j+1)*inc] + xv[(i+1)+(j+1)*inc])) / ac;

    xv[0+j*inc] = (bv[0+j*inc]
		   - ax * (ixfactor * xv[(nx-1)+j*inc] + xv[1+j*inc])
		   - ay * (xv[0+(j-1)*inc] + xv[0+(j+1)*inc])
		   - ad * (ixfactor * xv[(nx-1)+(j-1)*inc]
			   + xv[1+(j-1)*inc]
			   + ixfactor * xv[(nx-1)+(j+1)*inc]
			   + xv[1+(j+1)*inc])) / ac;
  }

#pragma omp single
  {
    xv[nx-1] = (bv[nx-1]
		- ax * (xv[nx-2] + xfactor * xv[0])
		- ay * (iyfactor * xv[(nx-1)+(ny-1)*inc] + xv[(nx-1)+inc])
		- ad * (iyfactor * xv[(nx-2)+(ny-1)*inc]
			+ xfactor * iyfactor * xv[0+(ny-1)*inc]
			+ xv[(nx-2)+inc]
			+ xfactor * xv[0+inc])) / ac;
    
    for(i=nx-2; i>0; i--)
      xv[i] = (bv[i]
	       - ax * (xv[i-1] + xv[i+1])
	       - ay * (iyfactor * xv[i+(ny-1)*inc] + xv[i+inc])
	       - ad * (iyfactor * xv[(i-1)+(ny-1)*inc]
		       + iyfactor * xv[(i+1)+(ny-1)*inc]
		       + xv[(i-1)+inc] + xv[(i+1)+inc])) / ac;

    xv[0] = (bv[0]
	     - ax * (ixfactor * xv[nx-1] + xv[1])
	     - ay * (iyfactor * xv[0+(ny-1)*inc] + xv[0+inc])
	     - ad * (ixfactor * iyfactor * xv[(nx-1)+(ny-1)*inc]
		     + iyfactor * xv[1+(ny-1)*inc]
		     + ixfactor * xv[(nx-1)+inc]
		     + xv[1+inc])) / ac;
  }
}

void
prolongation_node2d(field alpha, const node2d *c, node2d *f)
{
  const field *cv = c->v;
  field *fv = f->v;
  int cnx = c->gr->nx;
  int cny = c->gr->ny;
  int fnx = f->gr->nx;
  field xfactor = f->gr->xfactor;
  field yfactor = f->gr->yfactor;
  int cinc = cnx;
  int finc = fnx;
  int i, j;

  assert(f->gr->nx == 2 * c->gr->nx);
  assert(f->gr->ny == 2 * c->gr->ny);

#pragma omp for
  for(j=0; j<cny-1; j++) {
    for(i=0; i<cnx-1; i++) {
      fv[(2*i  )+ 2*j   *finc] += alpha * cv[i+j*cinc];
      fv[(2*i+1)+ 2*j   *finc] += 0.5 * alpha * (cv[i+j*cinc] +
						 cv[(i+1)+j*cinc]);
      fv[(2*i  )+(2*j+1)*finc] += 0.5 * alpha * (cv[i+j*cinc] +
						 cv[i+(j+1)*cinc]);
      fv[(2*i+1)+(2*j+1)*finc] += 0.25 * alpha * (cv[i+j*cinc] +
						  cv[(i+1)+j*cinc] +
						  cv[i+(j+1)*cinc] +
						  cv[(i+1)+(j+1)*cinc]);
    }

    fv[(2*i  )+ 2*j   *finc] += alpha * cv[i+j*cinc];
    fv[(2*i+1)+ 2*j   *finc] += 0.5 * alpha * (cv[i+j*cinc] +
					       xfactor * cv[0+j*cinc]);
    fv[(2*i  )+(2*j+1)*finc] += 0.5 * alpha * (cv[i+j*cinc] +
					       cv[i+(j+1)*cinc]);
    fv[(2*i+1)+(2*j+1)*finc] += 0.25 * alpha * (cv[i+j*cinc] +
						xfactor * cv[0+j*cinc] +
						cv[i+(j+1)*cinc] +
						xfactor * cv[0+(j+1)*cinc]);
  }

#pragma omp single
  {
    for(i=0; i<cnx-1; i++) {
      fv[(2*i  )+(2*cny-2)*finc] += alpha * cv[i+(cny-1)*cinc];
      fv[(2*i+1)+(2*cny-2)*finc] += 0.5 * alpha * (cv[i+(cny-1)*cinc] +
						   cv[(i+1)+(cny-1)*cinc]);
      fv[(2*i  )+(2*cny-1)*finc] += 0.5 * alpha * (cv[i+(cny-1)*cinc] +
						   yfactor * cv[i]);
      fv[(2*i+1)+(2*cny-1)*finc] += 0.25 * alpha * (cv[i+(cny-1)*cinc] +
						    cv[(i+1)+(cny-1)*cinc] +
						    yfactor * cv[i] +
						    yfactor * cv[i+1]);
    }

    fv[(2*i  )+(2*cny-2)*finc] += alpha * cv[i+(cny-1)*cinc];
    fv[(2*i+1)+(2*cny-2)*finc] += 0.5 * alpha * (cv[i+(cny-1)*cinc] +
						 xfactor * cv[0+(cny-1)*cinc]);
    fv[(2*i  )+(2*cny-1)*finc] += 0.5 * alpha * (cv[i+(cny-1)*cinc] +
						 yfactor * cv[i]);
    fv[(2*i+1)+(2*cny-1)*finc] += 0.25 * alpha * (cv[i+(cny-1)*cinc] +
						   xfactor * cv[0+(cny-1)*cinc] +
						   yfactor * cv[i] +
						   xfactor * yfactor * cv[0]);
  }
}

void
restriction_node2d(field alpha, const node2d *f, node2d *c)
{
  field *cv = c->v;
  const field *fv = f->v;
  int cnx = c->gr->nx;
  int cny = c->gr->ny;
  int fnx = f->gr->nx;
  int cinc = cnx;
  int finc = fnx;
  field ixfactor = CONJ(f->gr->xfactor);
  field iyfactor = CONJ(f->gr->yfactor);
  int i, j;

  assert(f->gr->nx == 2 * c->gr->nx);
  assert(f->gr->ny == 2 * c->gr->ny);

  /* Accumulate the first row */
#pragma omp single
  {
    cv[0] += alpha * (fv[0]
		      + 0.5 * ixfactor * fv[2*cnx-1]
		      + 0.5 * fv[1]
		      + 0.5 * iyfactor * fv[(2*cny-1)*finc]
		      + 0.5 * fv[finc]
		      + 0.25 * ixfactor * iyfactor * fv[(2*cnx-1)+(2*cny-1)*finc]
		      + 0.25 * iyfactor * fv[1+(2*cny-1)*finc]
		      + 0.25 * ixfactor * fv[(2*cny-1)+finc]
		      + 0.25 * fv[1+finc]);
    for(i=1; i<cnx; i++)
      cv[i] += alpha * (fv[2*i]
			+ 0.5 * fv[2*i-1]
			+ 0.5 * fv[2*i+1]
			+ 0.5 * iyfactor * fv[2*i+(2*cny-1)*finc]
			+ 0.5 * fv[2*i+finc]
			+ 0.25 * iyfactor * fv[(2*i-1)+(2*cny-1)*finc]
			+ 0.25 * iyfactor * fv[(2*i+1)+(2*cny-1)*finc]
			+ 0.25 * fv[(2*i-1)+finc]
			+ 0.25 * fv[(2*i+1)+finc]);
  }

  /* Accumulate the remaining rows */
#pragma omp for
  for(j=1; j<cny; j++) {
    cv[j*cinc] += alpha * (fv[(2*j)*finc]
			   + 0.5 * ixfactor * fv[(2*cnx-1)+(2*j)*finc]
			   + 0.5 * fv[1+(2*j)*finc]
			   + 0.5 * fv[(2*j-1)*finc]
			   + 0.5 * fv[(2*j+1)*finc]
			   + 0.25 * ixfactor * fv[(2*cnx-1)+(2*j-1)*finc]
			   + 0.25 * fv[1+(2*j-1)*finc]
			   + 0.25 * ixfactor * fv[(2*cnx-1)+(2*j+1)*finc]
			   + 0.25 * fv[1+(2*j+1)*finc]);
    
    for(i=1; i<cnx; i++)
      cv[i+j*cinc] += alpha * (fv[(2*i)+(2*j)*finc]
			       + 0.5 * fv[(2*i-1)+(2*j)*finc]
			       + 0.5 * fv[(2*i+1)+(2*j)*finc]
			       + 0.5 * fv[(2*i)+(2*j-1)*finc]
			       + 0.5 * fv[(2*i)+(2*j+1)*finc]
			       + 0.25 * fv[(2*i-1)+(2*j-1)*finc]
			       + 0.25 * fv[(2*i+1)+(2*j-1)*finc]
			       + 0.25 * fv[(2*i-1)+(2*j+1)*finc]
			       + 0.25 * fv[(2*i+1)+(2*j+1)*finc]);
  }
}

void
gradient_node2d(field alpha, const node2d *x, edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xv = x->v;
  field *yx = y->x + incx + 1;
  field *yy = y->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  int i, j;

  assert(y->gr == x->gr);

  /* Compute gradients along x edges */
#pragma omp for
  for(j=0; j<ny; j++) {
    for(i=0; i<nx-1; i++)
      yx[i+j*incx] += alpha * (xv[(i+1)+j*inc] - xv[i+j*inc]);

    yx[i+j*incx] += alpha * (xfactor * xv[0+j*inc] - xv[i+j*inc]);
  }

  /* Compute gradients along y edges */
#pragma omp for
  for(j=0; j<ny-1; j++)
    for(i=0; i<nx; i++)
      yy[i+j*incy] += alpha * (xv[i+(j+1)*inc] - xv[i+j*inc]);

#pragma omp single
  for(i=0; i<nx; i++)
    yy[i+(ny-1)*incy] += alpha * (yfactor * xv[i] - xv[i+(ny-1)*inc]);
}

void
adjgradient_node2d(field alpha, const edge2d *x, node2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  field *yv = y->v;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  int i, j;

  assert(y->gr == x->gr);

  /* Assemble gradients for the first line */
#pragma omp single
  {
    yv[0] += alpha * (CONJ(xfactor) * xx[nx-1] - xx[0]
		      + CONJ(yfactor) * xy[(ny-1)*incy] - xy[0]);
    
    for(i=1; i<nx; i++)
      yv[i] += alpha * (xx[i-1] - xx[i]
			+ CONJ(yfactor) * xy[i+(ny-1)*incy] - xy[i]);
  }

  /* Assemble gradients for the remaining lines */
#pragma omp for
  for(j=1; j<ny; j++) {
    yv[j*inc] += alpha * (CONJ(xfactor) * xx[(nx-1)+j*incx] - xx[j*incx]
			  + xy[(j-1)*incy] - xy[j*incy]);

    for(i=1; i<nx; i++)
      yv[i+j*inc] += alpha * (xx[(i-1)+j*incx] - xx[i+j*incx]
			      + xy[i+(j-1)*incy] - xy[i+j*incy]);
  }
}

void
vcycle_node2d(int L, int nu, const matrix *Ac,
	      node2d **b, node2d **x, node2d **d)
{
  int i, l;
  
  for(l=L; l>0; l--) {
    for(i=0; i<nu; i++)
      gsforward_node2d(1.0, b[l], x[l]);
    
    copy_node2d(b[l], d[l]);
    addeval_node2d(-1.0, x[l], d[l]);
    
    zero_node2d(b[l-1]);
    restriction_node2d(1.0, d[l], b[l-1]);
    
    zero_node2d(x[l-1]);
  }

  if(Ac) {
    assert(Ac->rows == x[0]->gr->nx * x[0]->gr->ny);

    copy_node2d(b[0], x[0]);
#pragma omp single
    potrsv_matrix(Ac, x[0]->v);
  }
  else {
    for(i=0; i<nu; i++)
      gssymm_node2d(1.0, b[0], x[0]);
  }
  
  for(l=1; l<=L; l++) {
    prolongation_node2d(1.0, x[l-1], x[l]);

    for(i=0; i<nu; i++)
      gsbackward_node2d(1.0, b[l], x[l]);
  }
}

void
mgcginit_node2d(int L, int nu, const node2d *b, node2d *x,
		node2d *p, node2d *a,
		node2d **r, node2d **q, node2d **d)
{
  copy_node2d(b, r[L]);
  addeval_node2d(-1.0, x, r[L]);

  zero_node2d(q[L]);
  vcycle_node2d(L, nu, 0, r, q, d);

  copy_node2d(q[L], p);
}

void
mgcgstep_node2d(int L, int nu, const node2d *b, node2d *x,
		node2d *p, node2d *a,
		node2d **r, node2d **q, node2d **d)
{
  field gamma, lambda, mu;
  
  zero_node2d(a);
  addeval_node2d(1.0, p, a);

  gamma = dotprod_node2d(p, a);
  lambda = dotprod_node2d(p, r[L]) / gamma;
  
  add_node2d(lambda, p, x);
  add_node2d(-lambda, a, r[L]);

  zero_node2d(q[L]);
  vcycle_node2d(L, nu, 0, r, q, d);

  mu = dotprod_node2d(a, q[L]) / gamma;
  scale_node2d(-mu, p);
  add_node2d(1.0, q[L], p);
}

void
densematrix_node2d(bool fixnull, const grid2d *gr, matrix *a)
{
  int nx = gr->nx;
  int ny = gr->ny;
  field *aa = a->a;
  int rows = a->rows;
  int cols = a->cols;
  int lda = a->ld;
  real hx = gr->hx;
  real hy = gr->hy;
  field xfactor = gr->xfactor;
  field yfactor = gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  field ac = 4.0 * (hx / hy + hy / hx) / 3.0;
  field ax = (hx / hy - 2.0 * hy / hx) / 3.0;
  field ay = (hy / hx - 2.0 * hx / hy) / 3.0;
  field ad = -(hx / hy + hy / hx) / 6.0;
  int i, j, ia, ja;

  assert(a->rows == a->cols);
  assert(a->rows == nx * ny);

  /* Reset the matrix */
  if(fixnull && ABS(xfactor-1.0) < 1e-8 && ABS(yfactor-1.0) < 1e-8) {
    for(j=0; j<cols; j++)
      for(i=0; i<rows; i++)
	aa[i+lda*j] = 0.1 / (rows * cols);
  }
  else {
    for(j=0; j<cols; j++)
      for(i=0; i<rows; i++)
	aa[i+lda*j] = 0.0;
  }

  /* Fill rows */
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      ia = i + j*nx;

      /* Diagonal */
      ja = ia;
      aa[ia+ja*lda] += ac;

      /* Left neighbour */
      if(i > 0) {
	ja = (i-1) + j*nx;
	aa[ia+ja*lda] += ax;
      }
      else {
	ja = (nx-1) + j*nx;
	aa[ia+ja*lda] += ixfactor * ax;
      }

      /* Right neighbour */
      if(i < nx-1) {
	ja = (i+1) + j*nx;
	aa[ia+ja*lda] += ax;
      }
      else {
	ja = 0 + j*nx;
	aa[ia+ja*lda] += xfactor * ax;
      }

      /* Lower neighbour */
      if(j > 0) {
	ja = i + (j-1)*nx;
	aa[ia+ja*lda] += ay;
      }
      else {
	ja = i + (ny-1)*nx;
	aa[ia+ja*lda] += iyfactor * ay;
      }

      /* Upper neighbour */
      if(j < ny-1) {
	ja = i + (j+1)*nx;
	aa[ia+ja*lda] += ay;
      }
      else {
	ja = i + 0*nx;
	aa[ia+ja*lda] += yfactor * ay;
      }

      /* Lower left neighbour */
      if(i > 0) {
	if(j > 0) {
	  ja = (i-1) + (j-1)*nx;
	  aa[ia+ja*lda] += ad;
	}
	else {
	  ja = (i-1) + (ny-1)*nx;
	  aa[ia+ja*lda] += iyfactor * ad;
	}
      }
      else {
	if(j > 0) {
	  ja = (nx-1) + (j-1)*nx;
	  aa[ia+ja*lda] += ixfactor * ad;
	}
	else {
	  ja = (nx-1) + (ny-1)*nx;
	  aa[ia+ja*lda] += ixfactor * iyfactor * ad;
	}
      }

      /* Lower right neighbour */
      if(i < nx-1) {
	if(j > 0) {
	  ja = (i+1) + (j-1)*nx;
	  aa[ia+ja*lda] += ad;
	}
	else {
	  ja = (i+1) + (ny-1)*nx;
	  aa[ia+ja*lda] += iyfactor * ad;
	}
      }
      else {
	if(j > 0) {
	  ja = 0 + (j-1)*nx;
	  aa[ia+ja*lda] += xfactor * ad;
	}
	else {
	  ja = 0 + (ny-1)*nx;
	  aa[ia+ja*lda] += xfactor * iyfactor * ad;
	}
      }

      /* Upper left neighbour */
      if(i > 0) {
	if(j < ny-1) {
	  ja = (i-1) + (j+1)*nx;
	  aa[ia+ja*lda] += ad;
	}
	else {
	  ja = (i-1) + 0*nx;
	  aa[ia+ja*lda] += yfactor * ad;
	}
      }
      else {
	if(j < ny-1) {
	  ja = (nx-1) + (j+1)*nx;
	  aa[ia+ja*lda] += ixfactor * ad;
	}
	else {
	  ja = (nx-1) + 0*nx;
	  aa[ia+ja*lda] += ixfactor * yfactor * ad;
	}
      }

      /* Upper right neighbour */
      if(i < nx-1) {
	if(j < ny-1) {
	  ja = (i+1) + (j+1)*nx;
	  aa[ia+ja*lda] += ad;
	}
	else {
	  ja = (i+1) + 0*nx;
	  aa[ia+ja*lda] += yfactor * ad;
	}
      }
      else {
	if(j < ny-1) {
	  ja = 0 + (j+1)*nx;
	  aa[ia+ja*lda] += xfactor * ad;
	}
	else {
	  ja = 0 + 0*nx;
	  aa[ia+ja*lda] += xfactor * yfactor * ad;
	}
      }
    }
}

void
denseto_node2d(const field *x, node2d *y)
{
  int nx = y->gr->nx;
  int ny = y->gr->ny;
  int inc = nx;
  field *yv = y->v;
  int i, j;

  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yv[i+j*inc] = x[i+j*nx];
}

void
densefrom_node2d(const node2d *x, field *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int inc = nx;
  const field *xv = x->v;
  int i, j;

  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      y[i+j*nx] = xv[i+j*inc];
}

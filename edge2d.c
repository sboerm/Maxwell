
#include "edge2d.h"

#include "basic.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#ifdef USE_CAIRO
#include "cairo.h"
#include "cairo-pdf.h"
#include "string.h"
#endif

/* DEBUGGING */
#include <stdio.h>

#define SMOOTHER_INTERLEAVED
#define SMOOTHER_LINES

edge2d *
new_edge2d(const grid2d *gr)
{
  edge2d *x;
  int nx = gr->nx;
  int ny = gr->ny;

  x = (edge2d *) malloc(sizeof(edge2d));
  x->gr = gr;
  x->x = (field *) malloc(sizeof(field) * (nx+1) * (ny+2));
  x->y = (field *) malloc(sizeof(field) * (nx+2) * (ny+1));

  return x;
}

void
del_edge2d(edge2d *x)
{
  free(x->y);
  x->y = 0;

  free(x->x);
  x->x = 0;

  x->gr = 0;

  free(x);
}

size_t
getdimension_edge2d(const edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;

  return (size_t) 2 * nx * ny;
}

size_t
getsize_edge2d(const edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  size_t sz;

  sz = sizeof(edge2d);
  sz += sizeof(field) * (nx+1) * (ny+2);
  sz += sizeof(field) * (nx+2) * (ny+1);
  
  return sz;
}

void
zero_edge2d(edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  int i, j;

  /* Clear coefficients for x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xx[i+j*incx] = 0.0;

  /* Clear coefficients for y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xy[i+j*incy] = 0.0;
}

void
copy_edge2d(const edge2d *x, edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  field *yx = y->x + incx + 1;
  field *yy = y->y + incy + 1;
  int i, j;

  assert(x->gr == y->gr);
  
  /* Copy x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yx[i+j*incx] = xx[i+j*incx];

  /* Copy y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yy[i+j*incy] = xy[i+j*incy];
}

void
swap_edge2d(edge2d *x, edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field *yx = y->x + incx + 1;
  field *yy = y->y + incy + 1;
  field swap;
  int i, j;

  assert(x->gr == y->gr);
  
  /* Swap x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      swap = yx[i+j*incx];
      yx[i+j*incx] = xx[i+j*incx];
      xx[i+j*incx] = swap;
    }

  /* Swap y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      swap = yy[i+j*incy];
      yy[i+j*incy] = xy[i+j*incy];
      xy[i+j*incy] = swap;
    }
}

void
random_edge2d(edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  int i, j;

  /* Set random coefficients for x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      xx[i+j*incx] = 2.0 * rand() / RAND_MAX - 1.0;
#ifdef FIELD_COMPLEX
      xx[i+j*incx] += I * (2.0 * rand() / RAND_MAX - 1.0);
#endif
    }

  /* Set random coefficients for y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      xy[i+j*incy] = 2.0 * rand() / RAND_MAX - 1.0;
#ifdef FIELD_COMPLEX
      xy[i+j*incy] += I * (2.0 * rand() / RAND_MAX - 1.0);
#endif
    }
}

#ifdef USE_CAIRO
void
cairodraw_edge2d(const edge2d *x, int gx, int gy,
		 const epspattern *pat, const char *filename)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  real hgx = (hx * nx) / gx;
  real hgy = (hy * ny) / gy;
  cairo_t *cr;
  cairo_surface_t *pdf, *png;
  cairo_matrix_t G;
  const char *c;
  int i, j, k, ig, jg;
  real xoff, yoff;
  field vx, vy;
  real px, py, tn;
  real len1, len2, maxlen, scale, xscale, y1scale, y2scale;

#pragma omp single
  {
    png = pdf = 0;
    
    for(c=filename; *c; c++)
      ;
    if(filename+4 <= c && strncmp(c-4, ".png", 4) == 0) {
      if(nx * hx >= ny * hy) {
	png = cairo_image_surface_create(CAIRO_FORMAT_RGB24, 1024, 1024.0 * (ny * hy) / (nx * hx));
	cr = cairo_create(png);

	cairo_rectangle(cr, 0, 0, 1024, 1024.0 * (ny * hy) / (nx * hx));
	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	cairo_fill(cr);
	
	cairo_scale(cr, 1024.0 / (nx * hx), 1024.0 / (nx * hx));
	cairo_set_line_width(cr, nx * hx * 0.002);
      }
      else {
	png = cairo_image_surface_create(CAIRO_FORMAT_RGB24, 1024.0 * (nx * hx) / (ny * hy), 1024);
	cr = cairo_create(png);

	cairo_rectangle(cr, 0, 0, 1024.0 * (nx * hx) / (ny * hy), 1024);
	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	cairo_fill(cr);
	
	cairo_scale(cr, 1024.0 / (ny * hy), 1024.0 / (ny * hy));
	cairo_set_line_width(cr, ny * hy * 0.002);
      }
    }
    else {
      if(nx * hx >= ny * hy) {
	pdf = cairo_pdf_surface_create(filename, 400.0, 400.0 * (ny * hy) / (nx * hx));
	cr = cairo_create(pdf);

	cairo_rectangle(cr, 0, 0, 400.0, 400.0 * (ny * hy) / (nx * hx));
	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	cairo_fill(cr);
	
	cairo_surface_destroy(pdf);
	cairo_scale(cr, 400.0 / (nx * hx), 400.0 / (nx * hx));
	cairo_set_line_width(cr, nx * hx * 0.002);
      }
      else {
	pdf = cairo_pdf_surface_create(filename, 400.0 * (nx * hx) / (ny * hy), 400.0);
	cr = cairo_create(pdf);

	cairo_rectangle(cr, 0, 0, 400.0 * (nx * hx) / (ny * hy), 400.0);
	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	cairo_fill(cr);
	
	cairo_surface_destroy(pdf);
	cairo_scale(cr, 400.0 / (ny * hy), 400.0 / (ny * hy));
	cairo_set_line_width(cr, ny * hy * 0.002);
      }
    }

    if(pat) {
      cairo_save(cr);
      
      cairo_set_line_width(cr, cairo_get_line_width(cr) * 3.0);
      cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);

      for(k=0; k<pat->rectangles; k++) {
	cairo_rectangle(cr, pat->rectangle[5*k+1], pat->rectangle[5*k+2],
			pat->rectangle[5*k+3], pat->rectangle[5*k+4]);
	cairo_stroke(cr);
      }

      for(k=0; k<pat->circles; k++) {
	cairo_arc(cr, pat->circle[4*k+1], pat->circle[4*k+2],
		  pat->circle[4*k+3], 0.0, 6.282);
	cairo_stroke(cr);
      }
      
      cairo_restore(cr);
    }
    
    maxlen = 0.0;
    for(jg=0; jg<gy; jg++) {
      j = (jg + 0.5) * ny / gy;
      yoff = (jg + 0.5) * hgy - j * hy;
      for(ig=0; ig<gx; ig++) {
	i = (ig + 0.5) * nx / gx;
	xoff = (ig + 0.5) * hgx - i * hx;
	
	vx = (1.0 - xoff) * xx[i+j*incx] + xoff * xx[(i+1)+j*incx];
	vy = (1.0 - yoff) * xy[i+j*incy] + yoff * xy[i+(j+1)*incy];
	
	len1 = sqrt(REAL(vx) * REAL(vx) + REAL(vy) * REAL(vy));
	len2 = sqrt(IMAG(vx) * IMAG(vx) + IMAG(vy) * IMAG(vy));
	if(len1 > maxlen)
	  maxlen = len1;
	if(len2 > maxlen)
	  maxlen = len2;
      }
    }

    scale = (hgx > hgy ? 0.5 * hgy / maxlen : 0.5 * hgx / maxlen);

    for(jg=0; jg<gy; jg++) {
      j = (jg + 0.5) * ny / gy;
      yoff = (jg + 0.5) * hgy - j * hy;
      for(ig=0; ig<gx; ig++) {
	i = (ig + 0.5) * nx / gx;
	xoff = (ig + 0.5) * hgx - i * hx;

	vx = (1.0 - xoff) * xx[i+j*incx] + xoff * xx[(i+1)+j*incx];
	vy = (1.0 - yoff) * xy[i+j*incy] + yoff * xy[i+(j+1)*incy];

	cairo_save(cr);

	/* Givens rotation to save work */
	px = REAL(vx);
	py = REAL(vy);
	if(fabs(px) > fabs(py)) {
	  tn = py / fabs(px);
	  G.xx = (px >= 0.0 ?
		  1.0 / sqrt(1.0 + tn * tn) :
		  -1.0 / sqrt(1.0 + tn * tn));
	  G.yx = tn / sqrt(1.0 + tn * tn);
	}
	else if(fabs(py) > 0.0) {
	  tn = px / fabs(py);
	  G.yx = (py >= 0.0 ?
		  1.0 / sqrt(1.0 + tn * tn) :
		  -1.0 / sqrt(1.0 + tn * tn));
	  G.xx = tn / sqrt(1.0 + tn * tn);
	}
	else {
	  G.xx = 1.0;
	  G.yx = 0.0;
	}
	G.xy = -G.yx;
	G.yy = G.xx;
	G.x0 = hgx * (ig+0.5);
	G.y0 = hgy * (jg+0.5);
	cairo_transform(cr, &G);

	/* Draw an arrow */
	xscale = maxlen * scale;
	y1scale = sqrt(px * px + py * py) * scale;
	y2scale = maxlen * scale;
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
	cairo_move_to(cr, -0.2 * xscale, -y1scale);
	cairo_line_to(cr,  0.2 * xscale, -y1scale);
	cairo_line_to(cr,  0.2 * xscale, 0.5 * y1scale);
	cairo_line_to(cr,  0.5 * xscale, 0.5 * y1scale);
	cairo_line_to(cr,           0.0, 0.5 * (y1scale + y2scale));
	cairo_line_to(cr, -0.5 * xscale, 0.5 * y1scale);
	cairo_line_to(cr, -0.2 * xscale, 0.5 * y1scale);
	cairo_close_path(cr);
	cairo_fill(cr);

	cairo_restore(cr);
      }
    }
    
    if(png) {
      cairo_surface_write_to_png(png, filename);
      cairo_surface_destroy(png);
    }
    cairo_destroy(cr);
  }
}
#endif

#ifdef USE_NETCDF
ncfile *
new_ncfile(const edge2d *x, int eigs, int blochsteps, int gx, int gy,
	   const epspattern *pat, const char *filename)
{
  ncfile *nf;
  int nc_rectangles, nc_rectangle, nc_circles, nc_circle;
  int dim[4];
  size_t start[1], size[1];
  int info;

  nf = (ncfile *) malloc(sizeof(ncfile));

  info = nc_create(filename, NC_CLOBBER | NC_64BIT_DATA, &nf->file);
  assert(info == NC_NOERR);

  info = nc_def_dim(nf->file, "blochsteps", blochsteps, &nf->nc_blochsteps);
  assert(info == NC_NOERR);
  nf->blochsteps = blochsteps;

  info = nc_def_dim(nf->file, "eigenvalues", eigs, &nf->nc_eigenvalues);
  assert(info == NC_NOERR);
  nf->eigenvalues = eigs;

  info = nc_def_dim(nf->file, "graphx", gx, &nf->nc_graphx);
  assert(info == NC_NOERR);
  nf->graphx = gx;

  info = nc_def_dim(nf->file, "graphy", gy, &nf->nc_graphy);
  assert(info == NC_NOERR);
  nf->graphy = gy;

  if(pat) {
    info = nc_def_dim(nf->file, "rectangles", 5*pat->rectangles, &nc_rectangles);
    assert(info == NC_NOERR);

    info = nc_def_dim(nf->file, "circles", 4*pat->circles, &nc_circles);
    assert(info == NC_NOERR);
  }
  else {
    info = nc_def_dim(nf->file, "rectangles", 0, &nc_rectangles);
    assert(info == NC_NOERR);

    info = nc_def_dim(nf->file, "circles", 0, &nc_circles);
    assert(info == NC_NOERR);
  }
  
  dim[0] = nf->nc_blochsteps;
  info = nc_def_var(nf->file, "xbloch", NC_DOUBLE, 1, dim, &nf->nc_xbloch);
  assert(info == NC_NOERR);
  
  dim[0] = nf->nc_blochsteps;
  info = nc_def_var(nf->file, "ybloch", NC_DOUBLE, 1, dim, &nf->nc_ybloch);
  assert(info == NC_NOERR);
  
  dim[0] = nf->nc_blochsteps;
  info = nc_def_var(nf->file, "xfactorr", NC_DOUBLE, 1, dim, &nf->nc_xfactorr);
  assert(info == NC_NOERR);
  
  dim[0] = nf->nc_blochsteps;
  info = nc_def_var(nf->file, "xfactori", NC_DOUBLE, 1, dim, &nf->nc_xfactori);
  assert(info == NC_NOERR);
  
  dim[0] = nf->nc_blochsteps;
  info = nc_def_var(nf->file, "yfactorr", NC_DOUBLE, 1, dim, &nf->nc_yfactorr);
  assert(info == NC_NOERR);
  
  dim[0] = nf->nc_blochsteps;
  info = nc_def_var(nf->file, "yfactori", NC_DOUBLE, 1, dim, &nf->nc_yfactori);
  assert(info == NC_NOERR);

  if(pat) {
    dim[0] = nc_rectangles;
    info = nc_def_var(nf->file, "rectangle", NC_DOUBLE, 1, dim, &nc_rectangle);
    assert(info == NC_NOERR);
    
    dim[0] = nc_circles;
    info = nc_def_var(nf->file, "circle", NC_DOUBLE, 1, dim, &nc_circle);
    assert(info == NC_NOERR);
  }
  
  dim[0] = nf->nc_blochsteps;
  dim[1] = nf->nc_eigenvalues;
  info = nc_def_var(nf->file, "lambda", NC_DOUBLE, 2, dim, &nf->nc_lambda);
  assert(info == NC_NOERR);

  dim[0] = nf->nc_blochsteps;
  dim[1] = nf->nc_eigenvalues;
  dim[2] = nf->nc_graphx;
  dim[3] = nf->nc_graphy;
  info = nc_def_var(nf->file, "vectorxr", NC_DOUBLE, 4, dim, &nf->nc_vectorxr);
  assert(info == NC_NOERR);

  dim[0] = nf->nc_blochsteps;
  dim[1] = nf->nc_eigenvalues;
  dim[2] = nf->nc_graphx;
  dim[3] = nf->nc_graphy;
  info = nc_def_var(nf->file, "vectorxi", NC_DOUBLE, 4, dim, &nf->nc_vectorxi);
  assert(info == NC_NOERR);

  dim[0] = nf->nc_blochsteps;
  dim[1] = nf->nc_eigenvalues;
  dim[2] = nf->nc_graphx;
  dim[3] = nf->nc_graphy;
  info = nc_def_var(nf->file, "vectoryr", NC_DOUBLE, 4, dim, &nf->nc_vectoryr);
  assert(info == NC_NOERR);

  dim[0] = nf->nc_blochsteps;
  dim[1] = nf->nc_eigenvalues;
  dim[2] = nf->nc_graphx;
  dim[3] = nf->nc_graphy;
  info = nc_def_var(nf->file, "vectoryi", NC_DOUBLE, 4, dim, &nf->nc_vectoryi);
  assert(info == NC_NOERR);

  info = nc_enddef(nf->file);
  assert(info == NC_NOERR);

  if(pat) {
    start[0] = 0;
    size[0] = 5 * pat->rectangles;
    info = nc_put_vara_double(nf->file, nc_rectangle, start, size,
			      pat->rectangle);
    assert(info == NC_NOERR);
    
    start[0] = 0;
    size[0] = 4 * pat->circles;
    info = nc_put_vara_double(nf->file, nc_circle, start, size,
			      pat->circle);
    assert(info == NC_NOERR);
  }
  
  return nf;
}

void
del_ncfile(ncfile *nf)
{
  int info;

  info = nc_close(nf->file);
  assert(info == NC_NOERR);
}

void
write_ncfile(ncfile *nf, int blochstep, const real *lambda,
	     const edge2d **eigs)
{
  int nx = eigs[0]->gr->nx;
  int ny = eigs[0]->gr->ny;
  real hx = eigs[0]->gr->hx;
  real hy = eigs[0]->gr->hy;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx, *xy;
  int gx = nf->graphx;
  int gy = nf->graphy;
  real hgx = (nx * hx) / gx;
  real hgy = (ny * hy) / gy;
  size_t index[4];
  int info;
  int i, j, k, ig, jg;
  double xoff, yoff;
  field vx, vy;
  double buf;

#pragma omp single
  {
    index[0] = blochstep;
    info = nc_put_var1_double(nf->file, nf->nc_xbloch, index, &eigs[0]->gr->xbloch);
    assert(info == NC_NOERR);
    
    index[0] = blochstep;
    info = nc_put_var1_double(nf->file, nf->nc_ybloch, index, &eigs[0]->gr->ybloch);
    assert(info == NC_NOERR);
    
    index[0] = blochstep;
    buf = REAL(eigs[0]->gr->xfactor);
    info = nc_put_var1_double(nf->file, nf->nc_xfactorr, index, &buf);
    assert(info == NC_NOERR);
    
    index[0] = blochstep;
    buf = IMAG(eigs[0]->gr->xfactor);
    info = nc_put_var1_double(nf->file, nf->nc_xfactori, index, &buf);
    assert(info == NC_NOERR);
    
    index[0] = blochstep;
    buf = REAL(eigs[0]->gr->yfactor);
    info = nc_put_var1_double(nf->file, nf->nc_yfactorr, index, &buf);
    assert(info == NC_NOERR);
    
    index[0] = blochstep;
    buf = IMAG(eigs[0]->gr->yfactor);
    info = nc_put_var1_double(nf->file, nf->nc_yfactori, index, &buf);
    assert(info == NC_NOERR);
    
    for(k=0; k<nf->eigenvalues; k++) {
      index[0] = blochstep;
      index[1] = k;
      buf = lambda[k] - 1.0;
      info = nc_put_var1_double(nf->file, nf->nc_lambda, index, &buf);
      assert(info == NC_NOERR);
    }
    
    for(k=0; k<nf->eigenvalues; k++) {
      assert(eigs[k]->gr == eigs[0]->gr);
      xx = eigs[k]->x + incx + 1;
      xy = eigs[k]->y + incy + 1;
      
      for(jg=0; jg<gy; jg++) {
	j = (jg + 0.5) * ny / gy;
	yoff = (jg + 0.5) * hgy - j * hy;
	
	for(ig=0; ig<gx; ig++) {
	  i = (ig + 0.5) * nx / gx;
	  xoff = (ig + 0.5) * hgx - i * hx;
	  
	  vx = (1.0 - xoff) * xx[i+j*incx] + xoff * xx[(i+1)+j*incx];
	  vy = (1.0 - yoff) * xy[i+j*incy] + yoff * xy[i+(j+1)*incy];
	  
	  index[0] = blochstep;
	  index[1] = k;
	  index[2] = ig;
	  index[3] = jg;
	  buf = REAL(vx);
	  info = nc_put_var1_double(nf->file, nf->nc_vectorxr, index, &buf);
	  assert(info == NC_NOERR);
	  
	  index[0] = blochstep;
	  index[1] = k;
	  index[2] = ig;
	  index[3] = jg;
	  buf = IMAG(vx);
	  info = nc_put_var1_double(nf->file, nf->nc_vectorxi, index, &buf);
	  assert(info == NC_NOERR);
	  
	  index[0] = blochstep;
	  index[1] = k;
	  index[2] = ig;
	  index[3] = jg;
	  buf = REAL(vy);
	  info = nc_put_var1_double(nf->file, nf->nc_vectoryr, index, &buf);
	  assert(info == NC_NOERR);
	  
	  index[0] = blochstep;
	  index[1] = k;
	  index[2] = ig;
	  index[3] = jg;
	  buf = IMAG(vy);
	  info = nc_put_var1_double(nf->file, nf->nc_vectoryi, index, &buf);
	  assert(info == NC_NOERR);
	}
      }
    }
  }
}
#endif

void
interpolate_edge2d(edge2d *x,
		   void (*func)(const real *x, field *fx, void *data),
		   void *data)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  real xv[2];
  field fx[2];
  int i, j;

  /* Interpolate on x edges */
#pragma omp for
  for(j=0; j<ny; j++) {
    xv[1] = j * hy;
    for(i=0; i<nx; i++) {
      xv[0] = (i + 0.5) * hx;
      func(xv, fx, data);
      xx[i+j*incx] = hx * fx[0];
    }
  }

  /* Interpolate on y edges */
#pragma omp for
  for(j=0; j<ny; j++) {
    xv[1] = (j + 0.5) * hy;
    for(i=0; i<nx; i++) {
      xv[0] = i * hx;
      func(xv, fx, data);
      xy[i+j*incy] = hy * fx[1];
    }
  }
}

void
l2functional_edge2d(edge2d *x, edge2d *xbuf, edge2d *ybuf,
		    void (*func)(const real *x, field *fx, void *data),
		    void *data)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field *xxbuf = xbuf->x + incx + 1;
  field *yxbuf = xbuf->y + incy + 1;
  field *xybuf = ybuf->x + incx + 1;
  field *yybuf = ybuf->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real xv[2];
  field fx[2];
  int i, j;

  assert(xbuf->gr == x->gr);
  assert(ybuf->gr == x->gr);
  
  /* Quadrature points on x edges */
#pragma omp for
  for(j=0; j<ny; j++) {
    xv[1] = j * hy;
    for(i=0; i<nx; i++) {
      xv[0] = (i + 0.5) * hx;

      func(xv, fx, data);

      xxbuf[i+j*incx] = fx[0];
      xybuf[i+j*incx] = fx[1];
    }
  }

  /* Quadrature points on y edges */
#pragma omp for
  for(j=0; j<ny; j++) {
    xv[1] = (j + 0.5) * hy;
    for(i=0; i<nx; i++) {
      xv[0] = i * hx;

      func(xv, fx, data);

      yxbuf[i+j*incy] = fx[0];
      yybuf[i+j*incy] = fx[1];
    }
  }

  /* Approximate integrals for x-edges */
#pragma omp single
  {
    for(i=0; i<nx-1; i++)
      xx[i] = hy * (0.5 * xxbuf[i]
		    + 0.125 * yxbuf[i]
		    + 0.125 * yxbuf[i+1]
		    + 0.125 * iyfactor * yxbuf[i+(ny-1)*incy]
		    + 0.125 * iyfactor * yxbuf[(i+1)+(ny-1)*incy]);

    xx[nx-1] = hy * (0.5 * xxbuf[nx-1]
		     + 0.125 * yxbuf[nx-1]
		     + 0.125 * xfactor * yxbuf[0]
		     + 0.125 * iyfactor * yxbuf[(nx-1)+(ny-1)*incy]
		     + 0.125 * xfactor * iyfactor * yxbuf[0+(ny-1)*incy]);
  }

#pragma omp for
  for(j=1; j<ny; j++) {
    for(i=0; i<nx-1; i++)
      xx[i+j*incx] = hy * (0.5 * xxbuf[i+j*incx]
			   + 0.125 * yxbuf[i+j*incy]
			   + 0.125 * yxbuf[(i+1)+j*incy]
			   + 0.125 * yxbuf[i+(j-1)*incy]
			   + 0.125 * yxbuf[(i+1)+(j-1)*incy]);

    xx[(nx-1)+j*incx] = hy * (0.5 * xxbuf[(nx-1)+j*incx]
			      + 0.125 * yxbuf[(nx-1)+j*incy]
			      + 0.125 * xfactor * yxbuf[0+j*incy]
			      + 0.125 * yxbuf[(nx-1)+(j-1)*incy]
			      + 0.125 * xfactor * yxbuf[0+(j-1)*incy]);
  }

  /* Approximate integrals for y-edges */
#pragma omp for
  for(j=0; j<ny-1; j++) {
    xy[0+j*incy] = hx * (0.5 * yybuf[0+j*incy]
			 + 0.125 * xybuf[0+j*incx]
			 + 0.125 * ixfactor * xybuf[(nx-1)+j*incx]
			 + 0.125 * xybuf[0+(j+1)*incx]
			 + 0.125 * ixfactor * xybuf[(nx-1)+(j+1)*incx]);

    for(i=1; i<nx; i++)
      xy[i+j*incy] = hx * (0.5 * yybuf[i+j*incy]
			   + 0.125 * xybuf[i   +  j   *incx]
			   + 0.125 * xybuf[i-1 +  j   *incx]
			   + 0.125 * xybuf[i   + (j+1)*incx]
			   + 0.125 * xybuf[i-1 + (j+1)*incx]);

  }

#pragma omp single
  {
    xy[0+(ny-1)*incy] = hx * (0.5 * yybuf[0+(ny-1)*incy]
			      + 0.125 * xybuf[0+(ny-1)*incx]
			      + 0.125 * ixfactor * xybuf[(nx-1)+(ny-1)*incx]
			      + 0.125 * yfactor * xybuf[0]
			      + 0.125 * ixfactor * yfactor * xybuf[nx-1]);

    for(i=1; i<nx; i++)
      xy[i+(ny-1)*incy] = hx * (0.5 * yybuf[i+(ny-1)*incy]
				+ 0.125 * xybuf[i+(ny-1)*incx]
				+ 0.125 * xybuf[(i-1)+(ny-1)*incx]
				+ 0.125 * yfactor * xybuf[i]
				+ 0.125 * yfactor * xybuf[i-1]);
  }
}
  
void
scale_edge2d(field alpha, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  int i, j;

  /* Scale x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xx[i+j*incx] *= alpha;

  /* Scale y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xy[i+j*incy] *= alpha;
}

void
add_edge2d(field alpha, const edge2d *x, edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  field *yx = y->x + incx + 1;
  field *yy = y->y + incy + 1;
  int i, j;

  assert(x->gr == y->gr);
  
  /* Add x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yx[i+j*incx] += alpha * xx[i+j*incx];

  /* Add y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      yy[i+j*incy] += alpha * xy[i+j*incy];
}

real
norm2_edge2d(const edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  real sum;
  int i, j;

  sum = 0.0;
  
  /* Add squares of x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += ABS2(xx[i+j*incx]);

  /* Add squares of y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += ABS2(xy[i+j*incy]);

  return sqrt(reduce_sum_real(sum));
}

real
normmax_edge2d(const edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  real amax, aval;
  int i, j;

  amax = 0.0;
  
  /* Maximum of x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      aval = ABS(xx[i+j*incx]);
      if(aval > amax)
	amax = aval;
    }

  /* Maximum of y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      aval = ABS(xy[i+j*incy]);
      if(aval > amax)
	amax = aval;
    }

  return reduce_max_real(amax);
}

real
l2norm_edge2d(const edge2d *x,
	      void (*func)(const real *x, field *fx, void *data),
	      void *data)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real xv[2];
  field fx[2], xval, yval;
  real sum;
  int i, j;

  sum = 0.0;

  /* Approximate integrals for the bottom line of x-edges */
#pragma omp single
  {
    xv[1] = 0.0;
    for(i=0; i<nx-1; i++) {
      xv[0] = (i + 0.5) * hx;

      /* Evaluate function (if not null) */
      if(func)
	func(xv, fx, data);
      else
	fx[0] = fx[1] = 0.0;

      /* Subtract x basis function */
      fx[0] -= xx[i] / hx;

      /* Subtract y basis functions of top box */
      fx[1] -= 0.5 * (xy[i] +
		      xy[i+1]) / hy;

      sum += 0.25 * (ABS2(fx[0]) + ABS2(fx[1])) * hx * hy;
    }

    xv[0] = (i + 0.5) * hx;
    
    /* Evaluate function (if not null) */
    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;
    
    /* Subtract x basis function */
    fx[0] -= xx[i] / hx;
    
    /* Subtract y basis functions of top box */
    fx[1] -= 0.5 * (xy[i] +
		    xfactor * xy[0]) / hy;
    
    sum += 0.25 * (ABS2(fx[0]) + ABS2(fx[1])) * hx * hy;
  }

  /* Approximate integrals for the interior x-edges */
#pragma omp for
  for(j=1; j<ny; j++) {
    xv[1] = j * hy;
    for(i=0; i<nx-1; i++) {
      xv[0] = (i + 0.5) * hx;

      /* Evaluate function (if not null) */
      if(func)
	func(xv, fx, data);
      else
	fx[0] = fx[1] = 0.0;

      /* Subtract x basis function */
      fx[0] -= xx[i+j*incx] / hx;
      sum += 0.5 * ABS2(fx[0]) * hx * hy;

      /* Subtract y basis functions of top box */
      yval = fx[1] - 0.5 * (xy[i  +j*incy] +
			    xy[i+1+j*incy]) / hy;
      sum += 0.25 * ABS2(yval) * hx * hy;

      /* Subtract y basis functions of bottom box */
      yval = fx[1] - 0.5 * (xy[i  +(j-1)*incy] +
			    xy[i+1+(j-1)*incy]) / hy;
      sum += 0.25 * ABS2(yval) * hx * hy;
    }

    xv[0] = (i + 0.5) * hx;
    
    /* Evaluate function (if not null) */
    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;
    
    /* Subtract x basis function */
    fx[0] -= xx[i+j*incx] / hx;
    sum += 0.5 * ABS2(fx[0]) * hx * hy;
    
    /* Subtract y basis functions of top box */
    yval = fx[1] - 0.5 * (xy[i+j*incy] +
			  xfactor * xy[0+j*incy]) / hy;
    sum += 0.25 * ABS2(yval) * hx * hy;
    
    /* Subtract y basis functions of bottom box */
    yval = fx[1] - 0.5 * (xy[i+(j-1)*incy] +
			  xfactor * xy[0+(j-1)*incy]) / hy;
    sum += 0.25 * ABS2(yval) * hx * hy;
  }

  /* Approximate integrals for the top line of x-edges */
#pragma omp single
  {
    j = ny;
    
    xv[1] = ny * hy;
    for(i=0; i<nx-1; i++) {
      xv[0] = (i + 0.5) * hx;

      /* Evaluate function (if not null) */
      if(func)
	func(xv, fx, data);
      else
	fx[0] = fx[1] = 0.0;

      /* Subtract x basis function */
      fx[0] -= iyfactor * xx[i] / hx;

      /* Subtract y basis functions of bottom box */
      fx[1] -= 0.5 * (xy[i+(j-1)*incy] +
		      xy[i+1+(j-1)*incy]) / hy;

      sum += 0.25 * (ABS2(fx[0]) + ABS2(fx[1])) * hx * hy;
    }
    
    xv[0] = (i + 0.5) * hx;
    
    /* Evaluate function (if not null) */
    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;
    
    /* Subtract x basis function */
    fx[0] -= iyfactor * xx[i] / hx;
    
    /* Subtract y basis functions of bottom box */
    fx[1] -= 0.5 * (xy[i+(j-1)*incy] +
		    xfactor * xy[0+(j-1)*incy]) / hy;
    
    sum += 0.25 * (ABS2(fx[0]) + ABS2(fx[1])) * hx * hy;
  }
  
  /* Approximate integral for the y-edges */
#pragma omp for
  for(j=0; j<ny-1; j++) {
    xv[1] = (j + 0.5) * hy;

    xv[0] = 0.0;

    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;

    /* Subtract y basis function */
    fx[1] -= xy[j*incy] / hy;
    sum += 0.25 * ABS2(fx[1]) * hx * hy;

    /* Subtract x basis functions for right box */
    fx[0] -= 0.5 * (xx[j*incx] +
		    xx[(j+1)*incx]) / hx;
    sum += 0.25 * ABS2(fx[0]) * hx * hy;

    for(i=1; i<nx; i++) {
      xv[0] = i * hx;

      if(func)
	func(xv, fx, data);
      else
	fx[0] = fx[1] = 0.0;

      /* Subtract y basis function */
      fx[1] -= xy[i+j*incy] / hy;
      sum += 0.5 * ABS2(fx[1]) * hx * hy;

      /* Subtract x basis functions for right box */
      xval = fx[0] - 0.5 * (xx[i+j*incx] +
			    xx[i+(j+1)*incx]) / hx;
      sum += 0.25 * ABS2(xval) * hx * hy;

      /* Subtract x basis functions for left box */
      xval = fx[0] - 0.5 * (xx[i-1+j*incx] +
			    xx[i-1+(j+1)*incx]) / hx;
      sum += 0.25 * ABS2(xval) * hx * hy;
    }

    xv[0] = i * hx;

    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;

    /* Subtract y basis function */
    fx[1] -= xfactor * xy[0+j*incy] / hy;
    sum += 0.25 * ABS2(fx[1]) * hx * hy;

    /* Subtract x basis functions for left box */
    fx[0] -= 0.5 * (xx[i-1+j*incx] +
		    xx[i-1+(j+1)*incx]) / hx;
    sum += 0.25 * ABS2(fx[0]) * hx * hy;
  }

#pragma omp single
  {
    j = ny-1;
    
    xv[1] = (j + 0.5) * hy;

    xv[0] = 0.0;

    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;

    /* Subtract y basis function */
    fx[1] -= xy[j*incy] / hy;
    sum += 0.25 * ABS2(fx[1]) * hx * hy;

    /* Subtract x basis functions for right box */
    fx[0] -= 0.5 * (xx[j*incx] +
		    yfactor * xx[0]) / hx;
    sum += 0.25 * ABS2(fx[0]) * hx * hy;

    for(i=1; i<nx; i++) {
      xv[0] = i * hx;

      if(func)
	func(xv, fx, data);
      else
	fx[0] = fx[1] = 0.0;

      /* Subtract y basis function */
      fx[1] -= xy[i+j*incy] / hy;
      sum += 0.5 * ABS2(fx[1]) * hx * hy;

      /* Subtract x basis functions for right box */
      xval = fx[0] - 0.5 * (xx[i+j*incx] +
			    yfactor * xx[i]) / hx;
      sum += 0.25 * ABS2(xval) * hx * hy;

      /* Subtract x basis functions for left box */
      xval = fx[0] - 0.5 * (xx[i-1+j*incx] +
			    yfactor * xx[i-1]) / hx;
      sum += 0.25 * ABS2(xval) * hx * hy;
    }

    xv[0] = i * hx;

    if(func)
      func(xv, fx, data);
    else
      fx[0] = fx[1] = 0.0;

    /* Subtract y basis function */
    fx[1] -= ixfactor * xy[0+j*incy] / hy;
    sum += 0.25 * ABS2(fx[1]) * hx * hy;

    /* Subtract x basis functions for left box */
    fx[0] -= 0.5 * (xx[i-1+j*incx] +
		    yfactor * xx[i-1]) / hx;
    sum += 0.25 * ABS2(fx[0]) * hx * hy;
  }
  
  return sqrt(reduce_sum_real(sum));
}

field
dotprod_edge2d(const edge2d *x, const edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  const field *yx = y->x + incx + 1;
  const field *yy = y->y + incy + 1;
  field sum;
  int i, j;

  assert(x->gr == y->gr);

  sum = 0.0;
  
  /* Add products for x edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += CONJ(xx[i+j*incx]) * yx[i+j*incx];

  /* Add products for y edges */
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += CONJ(xy[i+j*incy]) * yy[i+j*incy];

  return reduce_sum_field(sum);
}

void
nullprod_edge2d(const edge2d *x, field *xprod, field *yprod)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  real mx = hy / hx / 6.0;
  real my = hx / hy / 6.0;
  field sum;
  int i, j;

  assert(x->gr->xfactor == 1.0);
  assert(x->gr->yfactor == 1.0);
  
  /* Compute L^2 product of 1 + (yfactor-1) y and the x component */
  sum = 0.0;
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += 6.0 * mx * xx[i+j*incx];

  *xprod = reduce_sum_field(sum * hx);

  /* Compute L^2 product of 1 + (xfactor-1) x and the y component */
  sum = 0.0;
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      sum += 6.0 * my * xy[i+j*incy];

  *yprod = reduce_sum_field(sum * hy);
}

void
nulladd_edge2d(field xalpha, field yalpha, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  int i, j;

  assert(x->gr->xfactor == 1.0);
  assert(x->gr->yfactor == 1.0);
  
  xalpha *= hx;
  
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xx[i+j*incx] += xalpha;

  yalpha *= hy;
  
#pragma omp for
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      xy[i+j*incy] += yalpha;
}

void
center_edge2d(edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  real norm2, xlen, ylen;
  field xalpha, yalpha;

  assert(x->gr->xfactor == 1.0);
  assert(x->gr->yfactor == 1.0);
  
  nullprod_edge2d(x, &xalpha, &yalpha);

  xlen = nx * hx;
  ylen = ny * hy;
  
  norm2 = xlen * ylen;

  xalpha /= norm2;
  yalpha /= norm2;

  nulladd_edge2d(-xalpha, -yalpha, x);
}

void
addeval_edge2d(field alpha, field beta,
	       const edge2d *x, edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  field *yx = y->x + incx + 1;
  field *yy = y->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  field x0, x1, y0, y1;
  const real *eps = x->gr->eps;
  int i, j;
  
  assert(x->gr == y->gr);

#pragma omp single
  {
    /* Box (0,-1) */
    x0 = iyfactor * xx[0+(ny-1)*incx];
    x1 = xx[0];
    y0 = iyfactor * xy[0+(ny-1)*incy];
    y1 = iyfactor * xy[1+(ny-1)*incy];
    
    yx[0] += (axy * eps[0+(ny-1)*nx] * (x1 - x0 - y1 + y0)
	      + mx * (2.0 * x1 + x0));

    /* Box (0,0) */
    x0 = xx[0];
    x1 = xx[0+incx];
    y0 = xy[0];
    y1 = xy[1];
    
    yx[0]   += (axy * eps[0] * (x0 - x1 - y0 + y1)
		+ mx * (2.0 * x0 + x1));

    yy[0]   += (axy * eps[0] * (y0 - y1 - x0 + x1)
		+ my * (2.0 * y0 + y1));

    /* Box (-1,0) */
    x0 = ixfactor * xx[nx-1];
    x1 = ixfactor * xx[(nx-1)+incx];
    y0 = ixfactor * xy[nx-1];
    y1 = xy[0];
    
    yy[0] += (axy * eps[nx-1] * (y1 - y0 - x1 + x0)
	      + my * (2.0 * y1 + y0));

#ifdef USE_SIMD
#pragma omp simd
#endif
    for(i=1; i<nx-1; i++) {
      /* Box (i,-1) */
      x0 = iyfactor * xx[i+(ny-1)*incx];
      x1 = xx[i];
      y0 = iyfactor * xy[i+(ny-1)*incy];
      y1 = iyfactor * xy[(i+1)+(ny-1)*incy];
      
      yx[i] += (axy * eps[i+(ny-1)*nx] * (x1 - x0 - y1 + y0)
		+ mx * (2.0 * x1 + x0));

      /* Box (i,0) */
      x0 = xx[i];
      x1 = xx[i+incx];
      y0 = xy[i];
      y1 = xy[i+1];
      
      yx[i] += (axy * eps[i] * (x0 - x1 - y0 + y1)
		+ mx * (2.0 * x0 + x1));
      
      yy[i] += (axy * eps[i] * (y0 - y1 - x0 + x1)
		+ my * (2.0 * y0 + y1));

      /* Box (i-1,0) */
      x0 = xx[i-1];
      x1 = xx[(i-1)+incx];
      y0 = xy[i-1];
      y1 = xy[i];

      yy[i] += (axy * eps[i-1] * (y1 - y0 - x1 + x0)
		+ my * (2.0 * y1 + y0));
    }

    /* Box (nx-1,-1) */
    x0 = iyfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = xx[nx-1];
    y0 = iyfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xfactor * iyfactor * xy[0+(ny-1)*incy];
    
    yx[nx-1] += (axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 - y1 + y0)
		 + mx * (2.0 * x1 + x0));

    /* Box (nx-1,0) */
    x0 = xx[nx-1];
    x1 = xx[(nx-1)+incx];
    y0 = xy[nx-1];
    y1 = xfactor * xy[0];

    yx[nx-1] += (axy * eps[nx-1] * (x0 - x1 - y0 + y1)
		 + mx * (2.0 * x0 + x1));
    yy[nx-1] += (axy * eps[nx-1] * (y0 - y1 - x0 + x1)
		 + my * (2.0 * y0 + y1));

    /* Box (nx-2,0) */
    x0 = xx[nx-2];
    x1 = xx[(nx-2)+incx];
    y0 = xy[nx-2];
    y1 = xy[nx-1];

    yy[nx-1] += (axy * eps[nx-2] * (y1 - y0 - x1 + x0)
		 + my * (2.0 * y1 + y0));
  }
  
#pragma omp for
  for(j=1; j<ny-1; j++) {
    /* Box (0,j-1) */
    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    y0 = xy[0+(j-1)*incy];
    y1 = xy[1+(j-1)*incy];
      
    yx[0+j*incx] += (axy * eps[0+(j-1)*nx] * (x1 - x0 - y1 + y0)
		     + mx * (2.0 * x1 + x0));

    /* Box (0,j) */
    x0 = xx[0+j*incx];
    x1 = xx[0+(j+1)*incx];
    y0 = xy[0+j*incy];
    y1 = xy[1+j*incy];
    
    yx[0+j*incx] += (axy * eps[0+j*nx] * (x0 - x1 - y0 + y1)
		     + mx * (2.0 * x0 + x1));

    yy[0+j*incy] += (axy * eps[0+j*nx] * (y0 - y1 - x0 + x1)
		     + my * (2.0 * y0 + y1));

    /* Box (-1,j) */
    x0 = ixfactor * xx[(nx-1)+j*incx];
    x1 = ixfactor * xx[(nx-1)+(j+1)*incx];
    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
      
    yy[0+j*incy] += (axy * eps[(nx-1)+j*nx] * (y1 - y0 - x1 + x0)
		     + my * (2.0 * y1 + y0));

#ifdef USE_SIMD
#pragma omp simd
#endif
    for(i=1; i<nx-1; i++) {
      /* Box (i,j-1) */
      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      y0 = xy[i+(j-1)*incy];
      y1 = xy[(i+1)+(j-1)*incy];
      
      yx[i+j*incx] += (axy * eps[i+(j-1)*nx] * (x1 - x0 - y1 + y0)
		       + mx * (2.0 * x1 + x0));

      /* Box (i,j) */
      x0 = xx[i+j*incx];
      x1 = xx[i+(j+1)*incx];
      y0 = xy[i+j*incy];
      y1 = xy[(i+1)+j*incy];
      
      yx[i+j*incx] += (axy * eps[i+j*nx] * (x0 - x1 - y0 + y1)
		       + mx * (2.0 * x0 + x1));

      yy[i+j*incy] += (axy * eps[i+j*nx] * (y0 - y1 - x0 + x1)
		       + my * (2.0 * y0 + y1));

      /* Box (i-1,j) */
      x0 = xx[(i-1)+j*incx];
      x1 = xx[(i-1)+(j+1)*incx];
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      
      yy[i+j*incy] += (axy * eps[(i-1)+j*nx] * (y1 - y0 - x1 + x0)
		       + my * (2.0 * y1 + y0));
    }

    /* Box (nx-1,j-1) */
    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    y0 = xy[(nx-1)+(j-1)*incy];
    y1 = xfactor * xy[0+(j-1)*incy];
    
    yx[(nx-1)+j*incx] += (axy * eps[(nx-1)+j*nx] * (x1 - x0 - y1 + y0)
			  + mx * (2.0 * x1 + x0));

    /* Box (nx-1,j) */
    x0 = xx[(nx-1)+j*incx];
    x1 = xx[(nx-1)+(j+1)*incx];
    y0 = xy[(nx-1)+j*incy];
    y1 = xfactor * xy[0+j*incy];

    yx[(nx-1)+j*incx] += (axy * eps[(nx-1)+j*nx] * (x0 - x1 - y0 + y1)
			  + mx * (2.0 * x0 + x1));
    yy[(nx-1)+j*incy] += (axy * eps[(nx-1)+j*nx] * (y0 - y1 - x0 + x1)
			  + my * (2.0 * y0 + y1));

    /* Box (nx-2,j) */
    x0 = xx[(nx-2)+j*incx];
    x1 = xx[(nx-2)+(j+1)*incx];
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
      
    yy[(nx-1)+j*incy] += (axy * eps[(nx-2)+j*nx] * (y1 - y0 - x1 + x0)
			  + my * (2.0 * y1 + y0));
  }

#pragma omp single
  {
    /* Box (0,ny-2) */
    x0 = xx[0+(ny-2)*incx];
    x1 = xx[0+(ny-1)*incx];
    y0 = xy[0+(ny-2)*incy];
    y1 = xy[1+(ny-2)*incy];
      
    yx[0+(ny-1)*incx] += (axy * eps[0+(ny-1)*nx] * (x1 - x0 - y1 + y0)
			  + mx * (2.0 * x1 + x0));

    /* Box (0,ny-1) */
    x0 = xx[0+(ny-1)*incx];
    x1 = yfactor * xx[0];
    y0 = xy[0+(ny-1)*incy];
    y1 = xy[1+(ny-1)*incy];
      
    yx[0+(ny-1)*incx] += (axy * eps[0+(ny-1)*nx] * (x0 - x1 - y0 + y1)
			  + mx * (2.0 * x0 + x1));
    yy[0+(ny-1)*incy] += (axy * eps[0+(ny-1)*nx] * (y0 - y1 - x0 + x1)
			  + my * (2.0 * y0 + y1));

    /* Box (-1,ny-1) */
    x0 = ixfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = ixfactor * yfactor * xx[nx-1];
    y0 = ixfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xy[0+(ny-1)*incy];
      
    yy[0+(ny-1)*incy] += (axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y0 - x1 + x0)
			  + my * (2.0 * y1 + y0));

#ifdef USE_SIMD
#pragma omp simd
#endif
    for(i=1; i<nx-1; i++) {
      /* Box (i,ny-2) */
      x0 = xx[i+(ny-2)*incx];
      x1 = xx[i+(ny-1)*incx];
      y0 = xy[i+(ny-2)*incy];
      y1 = xy[(i+1)+(ny-2)*incy];
      
      yx[i+(ny-1)*incx] += (axy * eps[i+(ny-1)*nx] * (x1 - x0 - y1 + y0)
			    + mx * (2.0 * x1 + x0));

      /* Box (i,ny-1) */
      x0 = xx[i+(ny-1)*incx];
      x1 = yfactor * xx[i];
      y0 = xy[i+(ny-1)*incy];
      y1 = xy[(i+1)+(ny-1)*incy];
      
      yx[i+(ny-1)*incx] += (axy * eps[i+(ny-1)*nx] * (x0 - x1 - y0 + y1)
			    + mx * (2.0 * x0 + x1));
      yy[i+(ny-1)*incy] += (axy * eps[i+(ny-1)*nx] * (y0 - y1 - x0 + x1)
			    + my * (2.0 * y0 + y1));

      /* Box (i-1,ny-1) */
      x0 = xx[(i-1)+(ny-1)*incx];
      x1 = yfactor * xx[i-1];
      y0 = xy[(i-1)+(ny-1)*incy];
      y1 = xy[i+(ny-1)*incy];
      
      yy[i+(ny-1)*incy] += (axy * eps[(i-1)+(ny-1)*nx] * (y1 - y0 - x1 + x0)
			    + my * (2.0 * y1 + y0));
    }

    /* Box (nx-1,ny-2) */
    x0 = xx[(nx-1)+(ny-2)*incx];
    x1 = xx[(nx-1)+(ny-1)*incx];
    y0 = xy[(nx-1)+(ny-2)*incy];
    y1 = xfactor * xy[0+(ny-2)*incy];
    
    yx[(nx-1)+(ny-1)*incx] += (axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 - y1 + y0)
			       + mx * (2.0 * x1 + x0));

    /* Box (nx-1,ny-1) */
    x0 = xx[(nx-1)+(ny-1)*incx];
    x1 = yfactor * xx[nx-1];
    y0 = xy[(nx-1)+(ny-1)*incy];
    y1 = xfactor * xy[0+(ny-1)*incy];
    
    yx[(nx-1)+(ny-1)*incx] += (axy * eps[(nx-1)+(ny-1)*nx] * (x0 - x1 - y0 + y1)
			       + mx * (2.0 * x0 + x1));
    yy[(nx-1)+(ny-1)*incy] += (axy * eps[(nx-1)+(ny-1)*nx] * (y0 - y1 - x0 + x1)
			       + my * (2.0 * y0 + y1));

    /* Box (nx-2,ny-1) */
    x0 = xx[(nx-2)+(ny-1)*incx];
    x1 = yfactor * xx[nx-2];
    y0 = xy[(nx-2)+(ny-1)*incy];
    y1 = xy[(nx-1)+(ny-1)*incy];
    
    yy[(nx-1)+(ny-1)*incy] += (axy * eps[(nx-2)+(ny-1)*nx] * (y1 - y0 - x1 + x0)
			       + my * (2.0 * y1 + y0));
  }
}
      
field
energyprod_edge2d(field alpha, field beta,
		  const edge2d *x, const edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  const field *yx = y->x + incx + 1;
  const field *yy = y->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  field x0, x1, y0, y1;
  field sum;
  const real *eps = x->gr->eps;
  int i, j;
  
  assert(x->gr == y->gr);

  sum = 0.0;
  
#pragma omp single
  {
    /* Box (0,-1) */
    x0 = iyfactor * xx[0+(ny-1)*incx];
    x1 = xx[0];
    y0 = iyfactor * xy[0+(ny-1)*incy];
    y1 = iyfactor * xy[1+(ny-1)*incy];
    
    sum += CONJ(yx[0]) * (axy * eps[0+(ny-1)*nx] * (x1 - x0 - y1 + y0)
			  + mx * (2.0 * x1 + x0));

    /* Box (0,0) */
    x0 = xx[0];
    x1 = xx[0+incx];
    y0 = xy[0];
    y1 = xy[1];
    
    sum += CONJ(yx[0]) * (axy * eps[0] * (x0 - x1 - y0 + y1)
			  + mx * (2.0 * x0 + x1));
    sum += CONJ(yy[0]) * (axy * eps[0] * (y0 - y1 - x0 + x1)
			  + my * (2.0 * y0 + y1));

    /* Box (-1,0) */
    x0 = ixfactor * xx[nx-1];
    x1 = ixfactor * xx[(nx-1)+incx];
    y0 = ixfactor * xy[nx-1];
    y1 = xy[0];
    
    sum += CONJ(yy[0]) * (axy * eps[nx-1] * (y1 - y0 - x1 + x0)
			  + my * (2.0 * y1 + y0));

    for(i=1; i<nx-1; i++) {
      /* Box (i,-1) */
      x0 = iyfactor * xx[i+(ny-1)*incx];
      x1 = xx[i];
      y0 = iyfactor * xy[i+(ny-1)*incy];
      y1 = iyfactor * xy[(i+1)+(ny-1)*incy];
      
      sum += CONJ(yx[i]) * (axy * eps[i+(ny-1)*nx] * (x1 - x0 - y1 + y0)
			    + mx * (2.0 * x1 + x0));

      /* Box (i,0) */
      x0 = xx[i];
      x1 = xx[i+incx];
      y0 = xy[i];
      y1 = xy[i+1];
      
      sum += CONJ(yx[i]) * (axy * eps[i] * (x0 - x1 - y0 + y1)
			    + mx * (2.0 * x0 + x1));
      sum += CONJ(yy[i]) * (axy * eps[i] * (y0 - y1 - x0 + x1)
			    + my * (2.0 * y0 + y1));

      /* Box (i-1,0) */
      x0 = xx[i-1];
      x1 = xx[(i-1)+incx];
      y0 = xy[i-1];
      y1 = xy[i];

      sum += CONJ(yy[i]) * (axy * eps[i-1] * (y1 - y0 - x1 + x0)
			    + my * (2.0 * y1 + y0));
    }

    /* Box (nx-1,-1) */
    x0 = iyfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = xx[nx-1];
    y0 = iyfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xfactor * iyfactor * xy[0+(ny-1)*incy];
    
    sum += CONJ(yx[nx-1]) * (axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 - y1 + y0)
			     + mx * (2.0 * x1 + x0));

    /* Box (nx-1,0) */
    x0 = xx[nx-1];
    x1 = xx[(nx-1)+incx];
    y0 = xy[nx-1];
    y1 = xfactor * xy[0];

    sum += CONJ(yx[nx-1]) * (axy * eps[nx-1] * (x0 - x1 - y0 + y1)
			     + mx * (2.0 * x0 + x1));
    sum += CONJ(yy[nx-1]) * (axy * eps[nx-1] * (y0 - y1 - x0 + x1)
			     + my * (2.0 * y0 + y1));

    /* Box (nx-2,0) */
    x0 = xx[nx-2];
    x1 = xx[(nx-2)+incx];
    y0 = xy[nx-2];
    y1 = xy[nx-1];

    sum += CONJ(yy[nx-1]) * (axy * eps[nx-2] * (y1 - y0 - x1 + x0)
			     + my * (2.0 * y1 + y0));
  }
  
#pragma omp for
  for(j=1; j<ny-1; j++) {
    /* Box (0,j-1) */
    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    y0 = xy[0+(j-1)*incy];
    y1 = xy[1+(j-1)*incy];
      
    sum += CONJ(yx[0+j*incx]) * (axy * eps[0+(j-1)*nx] * (x1 - x0 - y1 + y0)
				 + mx * (2.0 * x1 + x0));

    /* Box (0,j) */
    x0 = xx[0+j*incx];
    x1 = xx[0+(j+1)*incx];
    y0 = xy[0+j*incy];
    y1 = xy[1+j*incy];
    
    sum += CONJ(yx[0+j*incx]) * (axy * eps[0+j*nx] * (x0 - x1 - y0 + y1)
				 + mx * (2.0 * x0 + x1));
    sum += CONJ(yy[0+j*incy]) * (axy * eps[0+j*nx] * (y0 - y1 - x0 + x1)
				 + my * (2.0 * y0 + y1));

    /* Box (-1,j) */
    x0 = ixfactor * xx[(nx-1)+j*incx];
    x1 = ixfactor * xx[(nx-1)+(j+1)*incx];
    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
      
    sum += CONJ(yy[0+j*incy]) * (axy * eps[(nx-1)+j*nx] * (y1 - y0 - x1 + x0)
				 + my * (2.0 * y1 + y0));

    for(i=1; i<nx-1; i++) {
      /* Box (i,j-1) */
      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      y0 = xy[i+(j-1)*incy];
      y1 = xy[(i+1)+(j-1)*incy];
      
      sum += CONJ(yx[i+j*incx]) * (axy * eps[i+(j-1)*nx] * (x1 - x0 - y1 + y0)
				   + mx * (2.0 * x1 + x0));

      /* Box (i,j) */
      x0 = xx[i+j*incx];
      x1 = xx[i+(j+1)*incx];
      y0 = xy[i+j*incy];
      y1 = xy[(i+1)+j*incy];
      
      sum += CONJ(yx[i+j*incx]) * (axy * eps[i+j*nx] * (x0 - x1 - y0 + y1)
				   + mx * (2.0 * x0 + x1));
      sum += CONJ(yy[i+j*incy]) * (axy * eps[i+j*nx] * (y0 - y1 - x0 + x1)
				   + my * (2.0 * y0 + y1));

      /* Box (i-1,j) */
      x0 = xx[(i-1)+j*incx];
      x1 = xx[(i-1)+(j+1)*incx];
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      
      sum += CONJ(yy[i+j*incy]) * (axy * eps[(i-1)+j*nx] * (y1 - y0 - x1 + x0)
				   + my * (2.0 * y1 + y0));
    }

    /* Box (nx-1,j-1) */
    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    y0 = xy[(nx-1)+(j-1)*incy];
    y1 = xfactor * xy[0+(j-1)*incy];
    
    sum += CONJ(yx[(nx-1)+j*incx]) * (axy * eps[(nx-1)+j*nx] * (x1 - x0 - y1 + y0)
				      + mx * (2.0 * x1 + x0));

    /* Box (nx-1,j) */
    x0 = xx[(nx-1)+j*incx];
    x1 = xx[(nx-1)+(j+1)*incx];
    y0 = xy[(nx-1)+j*incy];
    y1 = xfactor * xy[0+j*incy];

    sum += CONJ(yx[(nx-1)+j*incx]) * (axy * eps[(nx-1)+j*nx] * (x0 - x1 - y0 + y1)
				      + mx * (2.0 * x0 + x1));
    sum += CONJ(yy[(nx-1)+j*incy]) * (axy * eps[(nx-1)+j*nx] * (y0 - y1 - x0 + x1)
				      + my * (2.0 * y0 + y1));

    /* Box (nx-2,j) */
    x0 = xx[(nx-2)+j*incx];
    x1 = xx[(nx-2)+(j+1)*incx];
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
      
    sum += CONJ(yy[(nx-1)+j*incy]) * (axy * eps[(nx-2)+j*nx] * (y1 - y0 - x1 + x0)
				      + my * (2.0 * y1 + y0));
  }

#pragma omp single
  {
    /* Box (0,ny-2) */
    x0 = xx[0+(ny-2)*incx];
    x1 = xx[0+(ny-1)*incx];
    y0 = xy[0+(ny-2)*incy];
    y1 = xy[1+(ny-2)*incy];
      
    sum += CONJ(yx[0+(ny-1)*incx]) * (axy * eps[0+(ny-1)*nx] * (x1 - x0 - y1 + y0)
				      + mx * (2.0 * x1 + x0));

    /* Box (0,ny-1) */
    x0 = xx[0+(ny-1)*incx];
    x1 = yfactor * xx[0];
    y0 = xy[0+(ny-1)*incy];
    y1 = xy[1+(ny-1)*incy];
      
    sum += CONJ(yx[0+(ny-1)*incx]) * (axy * eps[0+(ny-1)*nx] * (x0 - x1 - y0 + y1)
				      + mx * (2.0 * x0 + x1));
    sum += CONJ(yy[0+(ny-1)*incy]) * (axy * eps[0+(ny-1)*nx] * (y0 - y1 - x0 + x1)
				      + my * (2.0 * y0 + y1));

    /* Box (-1,ny-1) */
    x0 = ixfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = ixfactor * yfactor * xx[nx-1];
    y0 = ixfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xy[0+(ny-1)*incy];
      
    sum += CONJ(yy[0+(ny-1)*incy]) * (axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y0 - x1 + x0)
				      + my * (2.0 * y1 + y0));

    for(i=1; i<nx-1; i++) {
      /* Box (i,ny-2) */
      x0 = xx[i+(ny-2)*incx];
      x1 = xx[i+(ny-1)*incx];
      y0 = xy[i+(ny-2)*incy];
      y1 = xy[(i+1)+(ny-2)*incy];
      
      sum += CONJ(yx[i+(ny-1)*incx]) * (axy * eps[i+(ny-1)*nx] * (x1 - x0 - y1 + y0)
					+ mx * (2.0 * x1 + x0));

      /* Box (i,ny-1) */
      x0 = xx[i+(ny-1)*incx];
      x1 = yfactor * xx[i];
      y0 = xy[i+(ny-1)*incy];
      y1 = xy[(i+1)+(ny-1)*incy];
      
      sum += CONJ(yx[i+(ny-1)*incx]) * (axy * eps[i+(ny-1)*nx] * (x0 - x1 - y0 + y1)
					+ mx * (2.0 * x0 + x1));
      sum += CONJ(yy[i+(ny-1)*incy]) * (axy * eps[i+(ny-1)*nx] * (y0 - y1 - x0 + x1)
					+ my * (2.0 * y0 + y1));

      /* Box (i-1,ny-1) */
      x0 = xx[(i-1)+(ny-1)*incx];
      x1 = yfactor * xx[i-1];
      y0 = xy[(i-1)+(ny-1)*incy];
      y1 = xy[i+(ny-1)*incy];
      
      sum += CONJ(yy[i+(ny-1)*incy]) * (axy * eps[(i-1)+(ny-1)*nx] * (y1 - y0 - x1 + x0)
					+ my * (2.0 * y1 + y0));
    }

    /* Box (nx-1,ny-2) */
    x0 = xx[(nx-1)+(ny-2)*incx];
    x1 = xx[(nx-1)+(ny-1)*incx];
    y0 = xy[(nx-1)+(ny-2)*incy];
    y1 = xfactor * xy[0+(ny-2)*incy];
    
    sum += CONJ(yx[(nx-1)+(ny-1)*incx]) * (axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 - y1 + y0)
					   + mx * (2.0 * x1 + x0));

    /* Box (nx-1,ny-1) */
    x0 = xx[(nx-1)+(ny-1)*incx];
    x1 = yfactor * xx[nx-1];
    y0 = xy[(nx-1)+(ny-1)*incy];
    y1 = xfactor * xy[0+(ny-1)*incy];
    
    sum += CONJ(yx[(nx-1)+(ny-1)*incx]) * (axy * eps[(nx-1)+(ny-1)*nx] * (x0 - x1 - y0 + y1)
					   + mx * (2.0 * x0 + x1));
    sum += CONJ(yy[(nx-1)+(ny-1)*incy]) * (axy * eps[(nx-1)+(ny-1)*nx] * (y0 - y1 - x0 + x1)
					   + my * (2.0 * y0 + y1));

    /* Box (nx-2,ny-1) */
    x0 = xx[(nx-2)+(ny-1)*incx];
    x1 = yfactor * xx[nx-2];
    y0 = xy[(nx-2)+(ny-1)*incy];
    y1 = xy[(nx-1)+(ny-1)*incy];
    
    sum += CONJ(yy[(nx-1)+(ny-1)*incy]) * (axy * eps[(nx-2)+(ny-1)*nx] * (y1 - y0 - x1 + x0)
						      + my * (2.0 * y1 + y0));
  }

  return reduce_sum_field(CONJ(sum));
}

field
massprod_edge2d(const edge2d *x, const edge2d *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  const field *yx = y->x + incx + 1;
  const field *yy = y->y + incy + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real mx = hy / hx / 6.0;
  real my = hx / hy / 6.0;
  field sum;
  int i, j;

  assert(x->gr == y->gr);

  sum = 0.0;

  /* Evaluate M for x edges */

#pragma omp single
  for(i=0; i<nx; i++)
    sum += CONJ(yx[i]) * mx * (4.0 * xx[i]
			       + iyfactor * xx[i+(ny-1)*incx]
			       + xx[i+incx]);
  
#pragma omp for
  for(j=1; j<ny-1; j++)
    for(i=0; i<nx; i++)
      sum += CONJ(yx[i+j*incx]) * mx * (4.0 * xx[i+j*incx]
					+ xx[i+(j-1)*incx]
					+ xx[i+(j+1)*incx]);

#pragma omp single
  for(i=0; i<nx; i++)
    sum += CONJ(yx[i+(ny-1)*incx]) * mx * (4.0 * xx[i+(ny-1)*incx]
					   + xx[i+(ny-2)*incx]
					   + yfactor * xx[i]);
  
  /* Evaluate M for y edges */

#pragma omp for
  for(j=0; j<ny; j++) {
    sum += CONJ(yy[0+j*incy]) * my * (4.0 * xy[0+j*incy]
					   + ixfactor * xy[(nx-1)+j*incy]
					   + xy[1+j*incy]);
    
    for(i=1; i<nx-1; i++)
      sum += CONJ(yy[i+j*incy]) * my * (4.0 * xy[i+j*incy]
					+ xy[(i-1)+j*incy]
					+ xy[(i+1)+j*incy]);
    
    sum += CONJ(yy[(nx-1)+j*incy]) * my * (4.0 * xy[(nx-1)+j*incy]
					   + xy[(nx-2)+j*incy]
					   + xfactor * xy[0+j*incy]);
  }
  
  return reduce_sum_field(CONJ(sum));
}

static void
starsolve_common(field axy, field mx, field my,
		 const field mxx[2][3], const field mxy[3][2],
		 const field mbx[2], const field mby[2],
		 const real meps[2][2],
		 field bl[4])
{
  real a, b, c, d;
  field Al[4][4], invdiag[4], qb[4];

  /* Set up the local transformed stiffness matrix */
  a = axy * meps[0][0];
  b = axy * meps[0][1];
  c = axy * meps[1][0];
  d = axy * meps[1][1];
  Al[0][0] = b + c;
  Al[1][0] = 0.0;
  Al[2][0] = b - c;
  Al[3][0] = 0.0;
  Al[1][1] = a + d;
  Al[2][1] = a - d;
  Al[3][1] = 0.0;
  Al[2][2] = a + b + c + d;

  /* Add the mass matrix */
  Al[0][0] += 4.0 * (mx + my);
  Al[1][0] += 4.0 * (mx - my);
  Al[2][2] += 4.0 * (mx + my);
  Al[3][2] = 4.0 * (mx - my);
  Al[3][3] = 4.0 * (mx + my);

  /* Compute the L D L^T factorization */
  invdiag[0] = 1.0 / Al[0][0];
  Al[1][0] *= invdiag[0];
  Al[2][0] *= invdiag[0];
  Al[1][1] -= Al[1][0] * Al[0][0] * CONJ(Al[1][0]);
  Al[2][1] -= Al[2][0] * Al[0][0] * CONJ(Al[1][0]);
  invdiag[1] = 1.0 / Al[1][1];
  Al[2][1] *= invdiag[1];
  Al[2][2] -= Al[2][1] * Al[1][1] * CONJ(Al[2][1]);
  invdiag[2] = 1.0 / Al[2][2];
  Al[3][2] *= invdiag[2];
  Al[3][3] -= Al[3][2] * Al[2][2] * CONJ(Al[3][2]);
  invdiag[3] = 1.0 / Al[3][3];
  
  /* Compute c = Q^* b */
  qb[0] = (mbx[0] + mbx[1] + mby[0] + mby[1]
	   - axy * meps[0][1] * (2.0 * mxx[0][1]
			  - 2.0 * mxx[0][2]
			  - 2.0 * mxy[0][1]
			  + 2.0 * mxy[1][1])
	   - axy * meps[1][0] * (2.0 * mxx[1][1]
			  - 2.0 * mxx[1][0]
			  + 2.0 * mxy[1][0]
			  - 2.0 * mxy[2][0])
	   - mx * (mxx[0][0]
		   + 4.0 * mxx[0][1]
		   + mxx[0][2])
	   - mx * (mxx[1][0]
		   + 4.0 * mxx[1][1]
		   + mxx[1][2])
	   - my * (mxy[0][0]
		   + 4.0 * mxy[1][0]
		   + mxy[2][0])
	   - my * (mxy[0][1]
		   + 4.0 * mxy[1][1]
		   + mxy[2][1]));
  qb[1] = (mbx[0] + mbx[1] - mby[0] - mby[1]
	   - axy * meps[0][0] * (2.0 * mxx[0][1]
			  - 2.0 * mxx[0][0]
			  + 2.0 * mxy[0][0]
			  - 2.0 * mxy[1][0])
	   - axy * meps[1][1] * (2.0 * mxx[1][1]
			  - 2.0 * mxx[1][2]
			  - 2.0 * mxy[1][1]
			  + 2.0 * mxy[2][1])
	   - mx * (mxx[0][0]
		   + 4.0 * mxx[0][1]
		   + mxx[0][2])
	   - mx * (mxx[1][0]
		   + 4.0 * mxx[1][1]
		   + mxx[1][2])
	   + my * (mxy[0][0]
		   + 4.0 * mxy[1][0]
		   + mxy[2][0])
	   + my * (mxy[0][1]
		   + 4.0 * mxy[1][1]
		   + mxy[2][1]));
  qb[2] = (mbx[0] - mbx[1] - mby[0] + mby[1]
	   - axy * meps[0][0] * (2.0 * mxx[0][1]
			  - 2.0 * mxx[0][0]
			  + 2.0 * mxy[0][0]
			  - 2.0 * mxy[1][0])
	   - axy * meps[0][1] * (2.0 * mxx[0][1]
			  - 2.0 * mxx[0][2]
			  - 2.0 * mxy[0][1]
			  + 2.0 * mxy[1][1])
	   + axy * meps[1][0] * (2.0 * mxx[1][1]
			  - 2.0 * mxx[1][0]
			  + 2.0 * mxy[1][0]
			  - 2.0 * mxy[2][0])
	   + axy * meps[1][1] * (2.0 * mxx[1][1]
			  - 2.0 * mxx[1][2]
			  - 2.0 * mxy[1][1]
			  + 2.0 * mxy[2][1])
	   - mx * (mxx[0][0]
		   + 4.0 * mxx[0][1]
		   + mxx[0][2])
	   + mx * (mxx[1][0]
		   + 4.0 * mxx[1][1]
		   + mxx[1][2])
	   + my * (mxy[0][0]
		   + 4.0 * mxy[1][0]
		   + mxy[2][0])
	   - my * (mxy[0][1]
		   + 4.0 * mxy[1][1]
		   + mxy[2][1]));
  qb[3] = (mbx[0] - mbx[1] + mby[0] - mby[1]
	   - mx * (mxx[0][0]
		   + 4.0 * mxx[0][1]
		   + mxx[0][2])
	   + mx * (mxx[1][0]
		   + 4.0 * mxx[1][1]
		   + mxx[1][2])
	   - my * (mxy[0][0]
		   + 4.0 * mxy[1][0]
		   + mxy[2][0])
	   + my * (mxy[0][1]
		   + 4.0 * mxy[1][1]
		   + mxy[2][1]));
  
  /* Forward solve */
  qb[1] -= Al[1][0] * qb[0];
  qb[2] -= Al[2][0] * qb[0];
  qb[2] -= Al[2][1] * qb[1];
  qb[3] -= Al[3][2] * qb[2];

  /* Diagonal solve */
  qb[0] *= invdiag[0];
  qb[1] *= invdiag[1];
  qb[2] *= invdiag[2];
  qb[3] *= invdiag[3];

  /* Backward solve */
  qb[2] -= CONJ(Al[3][2]) * qb[3];
  qb[0] -= CONJ(Al[2][0]) * qb[2];
  qb[1] -= CONJ(Al[2][1]) * qb[2];
  qb[0] -= CONJ(Al[1][0]) * qb[1];

  /* Compute b = 1/4 Q c */
  bl[0] = 0.25 * (qb[0] + qb[1] + qb[2] + qb[3]);
  bl[1] = 0.25 * (qb[0] + qb[1] - qb[2] - qb[3]);
  bl[2] = 0.25 * (qb[0] - qb[1] - qb[2] + qb[3]);
  bl[3] = 0.25 * (qb[0] - qb[1] + qb[2] - qb[3]);
}

static void
starsolve(int nx, int ny, int incx, int incy,
	  field axy, field mx, field my, const real *eps,
	  const field *bx, const field *by,
	  field *xx, field *xy,
	  int i, int j)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(i-1)+(j-1)*nx];
  meps[1][0] = eps[i+(j-1)*nx];
  meps[0][1] = eps[(i-1)+j*nx];
  meps[1][1] = eps[i+j*nx];

  /* Copy x values for the unknown x */
  mxx[0][0] = xx[(i-1)+(j-1)*incx];
  mxx[1][0] = xx[i+(j-1)*incx];
  mxx[0][1] = xx[(i-1)+j*incx];
  mxx[1][1] = xx[i+j*incx];
  mxx[0][2] = xx[(i-1)+(j+1)*incx];
  mxx[1][2] = xx[i+(j+1)*incx];

  /* Copy y values for the unknown x */
  mxy[0][0] = xy[(i-1)+(j-1)*incy];
  mxy[1][0] = xy[i+(j-1)*incy];
  mxy[2][0] = xy[(i+1)+(j-1)*incy];
  mxy[0][1] = xy[(i-1)+j*incy];
  mxy[1][1] = xy[i+j*incy];
  mxy[2][1] = xy[(i+1)+j*incy];

  /* Copy x values for the right-hand side b */
  mbx[0] = bx[(i-1)+j*incx];
  mbx[1] = bx[i+j*incx];

  /* Copy y values for the right-hand side b */
  mby[0] = by[i+(j-1)*incy];
  mby[1] = by[i+j*incy];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[(i-1)+j*incx] += bl[0];
  xx[i+j*incx] += bl[1];
  xy[i+(j-1)*incy] += bl[2];
  xy[i+j*incy] += bl[3];
}

/* Special version for i=0, j=0 */
static void
starsolve_lb(int nx, int ny, int incx, int incy,
	     field xfactor, field yfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(nx-1)+(ny-1)*nx];
  meps[1][0] = eps[(ny-1)*nx];
  meps[0][1] = eps[nx-1];
  meps[1][1] = eps[0];

  /* Copy x values for the unknown x */
  mxx[0][0] = CONJ(xfactor) * CONJ(yfactor) * xx[(nx-1)+(ny-1)*incx];
  mxx[1][0] = CONJ(yfactor) * xx[0+(ny-1)*incx];
  mxx[0][1] = CONJ(xfactor) * xx[nx-1];
  mxx[1][1] = xx[0];
  mxx[0][2] = CONJ(xfactor) * xx[(nx-1)+incx];
  mxx[1][2] = xx[0+incx];

  /* Copy y values for the unknown x */
  mxy[0][0] = CONJ(xfactor) * CONJ(yfactor) * xy[(nx-1)+(ny-1)*incy];
  mxy[1][0] = CONJ(yfactor) * xy[0+(ny-1)*incy];
  mxy[2][0] = CONJ(yfactor) * xy[1+(ny-1)*incy];
  mxy[0][1] = CONJ(xfactor) * xy[nx-1];
  mxy[1][1] = xy[0];
  mxy[2][1] = xy[1];

  /* Copy x values for the right-hand side b */
  mbx[0] = CONJ(xfactor) * bx[nx-1];
  mbx[1] = bx[0];

  /* Copy y values for the right-hand side b */
  mby[0] = CONJ(yfactor) * by[(ny-1)*incy];
  mby[1] = by[0];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[nx-1] += xfactor * bl[0];
  xx[0] += bl[1];
  xy[(ny-1)*incy] += yfactor * bl[2];
  xy[0] += bl[3];
}

/* Special version for i=nx-1, j=0 */
static void
starsolve_rb(int nx, int ny, int incx, int incy,
	     field xfactor, field yfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(nx-2)+(ny-1)*nx];
  meps[1][0] = eps[(nx-1)+(ny-1)*nx];
  meps[0][1] = eps[nx-2];
  meps[1][1] = eps[nx-1];

  /* Copy x values for the unknown x */
  mxx[0][0] = CONJ(yfactor) * xx[(nx-2)+(ny-1)*incx];
  mxx[1][0] = CONJ(yfactor) * xx[(nx-1)+(ny-1)*incx];
  mxx[0][1] = xx[nx-2];
  mxx[1][1] = xx[nx-1];
  mxx[0][2] = xx[(nx-2)+incx];
  mxx[1][2] = xx[(nx-1)+incx];

  /* Copy y values for the unknown x */
  mxy[0][0] = CONJ(yfactor) * xy[(nx-2)+(ny-1)*incy];
  mxy[1][0] = CONJ(yfactor) * xy[(nx-1)+(ny-1)*incy];
  mxy[2][0] = xfactor * CONJ(yfactor) * xy[0+(ny-1)*incy];
  mxy[0][1] = xy[nx-2];
  mxy[1][1] = xy[nx-1];
  mxy[2][1] = xfactor * xy[0];

  /* Copy x values for the right-hand side b */
  mbx[0] = bx[nx-2];
  mbx[1] = bx[nx-1];

  /* Copy y values for the right-hand side b */
  mby[0] = CONJ(yfactor) * by[(nx-1)+(ny-1)*incy];
  mby[1] = by[nx-1];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[nx-2] += bl[0];
  xx[nx-1] += bl[1];
  xy[(nx-1)+(ny-1)*incy] += yfactor * bl[2];
  xy[nx-1] += bl[3];
}

/* Special version for i=0, j=ny-1 */
static void
starsolve_lu(int nx, int ny, int incx, int incy,
	     field xfactor, field yfactor, 
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(nx-1)+(ny-2)*nx];
  meps[1][0] = eps[(ny-2)*nx];
  meps[0][1] = eps[(nx-1)+(ny-1)*nx];
  meps[1][1] = eps[(ny-1)*nx];

  /* Copy x values for the unknown x */
  mxx[0][0] = CONJ(xfactor) * xx[(nx-1)+(ny-2)*incx];
  mxx[1][0] = xx[0+(ny-2)*incx];
  mxx[0][1] = CONJ(xfactor) * xx[(nx-1)+(ny-1)*incx];
  mxx[1][1] = xx[0+(ny-1)*incx];
  mxx[0][2] = CONJ(xfactor) * yfactor * xx[nx-1];
  mxx[1][2] = yfactor * xx[0];

  /* Copy y values for the unknown x */
  mxy[0][0] = CONJ(xfactor) * xy[(nx-1)+(ny-2)*incy];
  mxy[1][0] = xy[0+(ny-2)*incy];
  mxy[2][0] = xy[1+(ny-2)*incy];
  mxy[0][1] = CONJ(xfactor) * xy[(nx-1)+(ny-1)*incy];
  mxy[1][1] = xy[0+(ny-1)*incy];
  mxy[2][1] = xy[1+(ny-1)*incy];

  /* Copy x values for the right-hand side b */
  mbx[0] = CONJ(xfactor) * bx[(nx-1)+(ny-1)*incx];
  mbx[1] = bx[0 + (ny-1)*incx];

  /* Copy y values for the right-hand side b */
  mby[0] = by[0+(ny-2)*incy];
  mby[1] = by[0+(ny-1)*incy];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[(nx-1)+(ny-1)*incx] += xfactor * bl[0];
  xx[0+(ny-1)*incx] += bl[1];
  xy[0+(ny-2)*incy] += bl[2];
  xy[0+(ny-1)*incy] += bl[3];
}

/* Special version for i=nx-1, j=ny-1 */
static void
starsolve_ru(int nx, int ny, int incx, int incy,
	     field xfactor, field yfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(nx-2)+(ny-2)*nx];
  meps[1][0] = eps[(nx-1)+(ny-2)*nx];
  meps[0][1] = eps[(nx-2)+(ny-1)*nx];
  meps[1][1] = eps[(nx-1)+(ny-1)*nx];

  /* Copy x values for the unknown x */
  mxx[0][0] = xx[(nx-2)+(ny-2)*incx];
  mxx[1][0] = xx[(nx-1)+(ny-2)*incx];
  mxx[0][1] = xx[(nx-2)+(ny-1)*incx];
  mxx[1][1] = xx[(nx-1)+(ny-1)*incx];
  mxx[0][2] = yfactor * xx[nx-2];
  mxx[1][2] = yfactor * xx[nx-1];

  /* Copy y values for the unknown x */
  mxy[0][0] = xy[(nx-2)+(ny-2)*incy];
  mxy[1][0] = xy[(nx-1)+(ny-2)*incy];
  mxy[2][0] = xfactor * xy[0+(ny-2)*incy];
  mxy[0][1] = xy[(nx-2)+(ny-1)*incy];
  mxy[1][1] = xy[(nx-1)+(ny-1)*incy];
  mxy[2][1] = xfactor * xy[0+(ny-1)*incy];

  /* Copy x values for the right-hand side b */
  mbx[0] = bx[(nx-2)+(ny-1)*incx];
  mbx[1] = bx[(nx-1)+(ny-1)*incx];

  /* Copy y values for the right-hand side b */
  mby[0] = by[(nx-1)+(ny-2)*incy];
  mby[1] = by[(nx-1)+(ny-1)*incy];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[(nx-2)+(ny-1)*incx] += bl[0];
  xx[(nx-1)+(ny-1)*incx] += bl[1];
  xy[(nx-1)+(ny-2)*incy] += bl[2];
  xy[(nx-1)+(ny-1)*incy] += bl[3];
}

/* Special version for i in [1:nx-2], j=0 */
static void
starsolve_mb(int nx, int ny, int incx, int incy,
	     field yfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy,
	     int i)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(i-1)+(ny-1)*nx];
  meps[1][0] = eps[i+(ny-1)*nx];
  meps[0][1] = eps[i-1];
  meps[1][1] = eps[i];

  /* Copy x values for the unknown x */
  mxx[0][0] = CONJ(yfactor) * xx[(i-1)+(ny-1)*incx];
  mxx[1][0] = CONJ(yfactor) * xx[i+(ny-1)*incx];
  mxx[0][1] = xx[i-1];
  mxx[1][1] = xx[i];
  mxx[0][2] = xx[(i-1)+incx];
  mxx[1][2] = xx[i+incx];

  /* Copy y values for the unknown x */
  mxy[0][0] = CONJ(yfactor) * xy[(i-1)+(ny-1)*incy];
  mxy[1][0] = CONJ(yfactor) * xy[i+(ny-1)*incy];
  mxy[2][0] = CONJ(yfactor) * xy[(i+1)+(ny-1)*incy];
  mxy[0][1] = xy[i-1];
  mxy[1][1] = xy[i];
  mxy[2][1] = xy[i+1];

  /* Copy x values for the right-hand side b */
  mbx[0] = bx[i-1];
  mbx[1] = bx[i];

  /* Copy y values for the right-hand side b */
  mby[0] = CONJ(yfactor) * by[i+(ny-1)*incy];
  mby[1] = by[i];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[i-1] += bl[0];
  xx[i] += bl[1];
  xy[i+(ny-1)*incy] += yfactor * bl[2];
  xy[i] += bl[3];
}

/* Special version for i in [1:nx-2], j=ny-1 */
static void
starsolve_mu(int nx, int ny, int incx, int incy,
	     field yfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy,
	     int i)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(i-1)+(ny-2)*nx];
  meps[1][0] = eps[i+(ny-2)*nx];
  meps[0][1] = eps[(i-1)+(ny-1)*nx];
  meps[1][1] = eps[i+(ny-1)*nx];

  /* Copy x values for the unknown x */
  mxx[0][0] = xx[(i-1)+(ny-2)*incx];
  mxx[1][0] = xx[i+(ny-2)*incx];
  mxx[0][1] = xx[(i-1)+(ny-1)*incx];
  mxx[1][1] = xx[i+(ny-1)*incx];
  mxx[0][2] = yfactor * xx[i-1];
  mxx[1][2] = yfactor * xx[i];

  /* Copy y values for the unknown x */
  mxy[0][0] = xy[(i-1)+(ny-2)*incy];
  mxy[1][0] = xy[i+(ny-2)*incy];
  mxy[2][0] = xy[(i+1)+(ny-2)*incy];
  mxy[0][1] = xy[(i-1)+(ny-1)*incy];
  mxy[1][1] = xy[i+(ny-1)*incy];
  mxy[2][1] = xy[(i+1)+(ny-1)*incy];

  /* Copy x values for the right-hand side b */
  mbx[0] = bx[(i-1)+(ny-1)*incx];
  mbx[1] = bx[i+(ny-1)*incx];

  /* Copy y values for the right-hand side b */
  mby[0] = by[i+(ny-2)*incy];
  mby[1] = by[i+(ny-1)*incy];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[(i-1)+(ny-1)*incx] += bl[0];
  xx[i+(ny-1)*incx] += bl[1];
  xy[i+(ny-2)*incy] += bl[2];
  xy[i+(ny-1)*incy] += bl[3];
}

/* Special version for i=0, j in [1:ny-2] */
static void
starsolve_lm(int nx, int ny, int incx, int incy,
	     field xfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy,
	     int j)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(nx-1)+(j-1)*nx];
  meps[1][0] = eps[0+(j-1)*nx];
  meps[0][1] = eps[(nx-1)+j*nx];
  meps[1][1] = eps[0+j*nx];

  /* Copy x values for the unknown x */
  mxx[0][0] = CONJ(xfactor) * xx[(nx-1)+(j-1)*incx];
  mxx[1][0] = xx[0+(j-1)*incx];
  mxx[0][1] = CONJ(xfactor) * xx[(nx-1)+j*incx];
  mxx[1][1] = xx[0+j*incx];
  mxx[0][2] = CONJ(xfactor) * xx[(nx-1)+(j+1)*incx];
  mxx[1][2] = xx[0+(j+1)*incx];

  /* Copy y values for the unknown x */
  mxy[0][0] = CONJ(xfactor) * xy[(nx-1)+(j-1)*incy];
  mxy[1][0] = xy[0+(j-1)*incy];
  mxy[2][0] = xy[1+(j-1)*incy];
  mxy[0][1] = CONJ(xfactor) * xy[(nx-1)+j*incy];
  mxy[1][1] = xy[0+j*incy];
  mxy[2][1] = xy[1+j*incy];

  /* Copy x values for the right-hand side b */
  mbx[0] = CONJ(xfactor) * bx[(nx-1)+j*incx];
  mbx[1] = bx[0+j*incx];

  /* Copy y values for the right-hand side b */
  mby[0] = by[0+(j-1)*incy];
  mby[1] = by[0+j*incy];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[(nx-1)+j*incx] += xfactor * bl[0];
  xx[0+j*incx] += bl[1];
  xy[0+(j-1)*incy] += bl[2];
  xy[0+j*incy] += bl[3];
}

/* Special case i=nx-1, j in [1:ny-2] */
static void
starsolve_rm(int nx, int ny, int incx, int incy,
	     field xfactor,
	     field axy, field mx, field my, const real *eps,
	     const field *bx, const field *by,
	     field *xx, field *xy,
	     int j)
{
  real meps[2][2];
  field mxx[2][3], mxy[3][2], mbx[2], mby[2];
  field bl[4];

  /* Copy eps values */
  meps[0][0] = eps[(nx-2)+(j-1)*nx];
  meps[1][0] = eps[(nx-1)+(j-1)*nx];
  meps[0][1] = eps[(nx-2)+j*nx];
  meps[1][1] = eps[(nx-1)+j*nx];

  /* Copy x values for the unknown x */
  mxx[0][0] = xx[(nx-2)+(j-1)*incx];
  mxx[1][0] = xx[(nx-1)+(j-1)*incx];
  mxx[0][1] = xx[(nx-2)+j*incx];
  mxx[1][1] = xx[(nx-1)+j*incx];
  mxx[0][2] = xx[(nx-2)+(j+1)*incx];
  mxx[1][2] = xx[(nx-1)+(j+1)*incx];

  /* Copy y values for the unknown x */
  mxy[0][0] = xy[(nx-2)+(j-1)*incy];
  mxy[1][0] = xy[(nx-1)+(j-1)*incy];
  mxy[2][0] = xfactor * xy[0+(j-1)*incy];
  mxy[0][1] = xy[(nx-2)+j*incy];
  mxy[1][1] = xy[(nx-1)+j*incy];
  mxy[2][1] = xfactor * xy[0+j*incy];

  /* Copy x values for the right-hand side b */
  mbx[0] = bx[(nx-2)+j*incx];
  mbx[1] = bx[(nx-1)+j*incx];

  /* Copy y values for the right-hand side b */
  mby[0] = by[(nx-1)+(j-1)*incy];
  mby[1] = by[(nx-1)+j*incy];

  /* Compute update */
  starsolve_common(axy, mx, my, mxx, mxy, mbx, mby, meps, bl);

  /* Write new values */
  xx[(nx-2)+j*incx] += bl[0];
  xx[(nx-1)+j*incx] += bl[1];
  xy[(nx-1)+(j-1)*incy] += bl[2];
  xy[(nx-1)+j*incy] += bl[3];
}

#ifdef SMOOTHER_LINES
void
gsforward_edge2d(field alpha, field beta,
		 const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  int i, j;

  assert(x->gr == b->gr);

  /* Bottom line */
#pragma omp single
  {
    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=1; i<nx-1; i++)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_rb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i++)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i++)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }
  
  /* Top line */
#pragma omp single
  {
    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=1; i<nx-1; i++)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_ru(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }
}

void
gsbackward_edge2d(field alpha, field beta,
		  const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  int i, j;

  assert(x->gr == b->gr);

  /* Top line */
#pragma omp single
  {
    starsolve_ru(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=nx-2; i>0; i--)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=nx-2; i>0; i--)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }
  
  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=nx-2; i>0; i--)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* Bottom line */
#pragma omp single
  {
    starsolve_rb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=nx-2; i>0; i--)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }
}

void
gssymm_edge2d(field alpha, field beta,
	      const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  int i, j;

  assert(x->gr == b->gr);

  /* Bottom line */
#pragma omp single
  {
    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=1; i<nx-1; i++)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_rb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i++)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i++)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }
  
  /* Top line */
#pragma omp single
  {
    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=1; i<nx-1; i++)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_ru(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Top line */
#pragma omp single
  {
    starsolve_ru(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=nx-2; i>0; i--)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=nx-2; i>0; i--)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }
  
  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    starsolve_rm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=nx-2; i>0; i--)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* Bottom line */
#pragma omp single
  {
    starsolve_rb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=nx-2; i>0; i--)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }
}
#else
void
gsforward_edge2d(field alpha, field beta,
		 const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  int i, j;

  assert(x->gr == b->gr);

  /* ---- Even rows, even columns */
  
  /* Bottom line, even columns */
#pragma omp single
  {
    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 0)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* Even rows, even columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Even rows, odd columns */
  
  /* Bottom line, odd columns */
#pragma omp single
  {
    for(i=1; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 1)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* Even rows, odd columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    for(i=1; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Odd rows, even columns */
  
  /* Odd rows, even columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Odd rows, odd columns */
  
  /* Odd rows, odd columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    for(i=1; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }
}

void
gsbackward_edge2d(field alpha, field beta,
		  const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  int i, j;

  assert(x->gr == b->gr);

  /* ---- Odd rows, odd columns */
  
  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=1; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);
  }

  /* Odd rows, odd columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* ---- Odd rows, even columns */
  
  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Odd rows, even columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* ---- Even rows, odd columns */
  
  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=1; i+1<nx; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);
  }

  /* Even rows, odd columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);

    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);
  }

  /* Bottom line, odd columns */
#pragma omp single
  {
    if((nx-1) % 2 == 1)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=1; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);
  }

  /* ---- Even rows, even columns */
  
  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Even rows, even columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* Bottom line, even columns */
#pragma omp single
  {
    if((nx-1) % 2 == 0)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }
}

void
gssymm_edge2d(field alpha, field beta,
	      const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  int i, j;

  assert(x->gr == b->gr);

  /* ---- Even rows, even columns */
  
  /* Bottom line, even columns */
#pragma omp single
  {
    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 0)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* Even rows, even columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Even rows, odd columns */
  
  /* Bottom line, odd columns */
#pragma omp single
  {
    for(i=1; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 1)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* Even rows, odd columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    for(i=1; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Odd rows, even columns */
  
  /* Odd rows, even columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Odd rows, odd columns */
  
  /* Odd rows, odd columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
  }

  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    for(i=1; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);
  }

  /* ---- Odd rows, odd columns */
  
  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    for(i=1; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);
  }

  /* Odd rows, odd columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
    
    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* ---- Odd rows, even columns */
  
  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 1) {
    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Odd rows, even columns */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* ---- Even rows, odd columns */
  
  /* Top line, odd columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    if((nx-1) % 2 == 1)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=1; i+1<nx; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);
  }

  /* Even rows, odd columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    if((nx-1) % 2 == 1)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);

    for(i=1; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);
  }

  /* Bottom line, odd columns */
#pragma omp single
  {
    if((nx-1) % 2 == 1)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=1; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);
  }

  /* ---- Even rows, even columns */
  
  /* Top line, even columns */
#pragma omp single
  if((ny-1) % 2 == 0) {
    if((nx-1) % 2 == 0)
      starsolve_ru(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mu(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lu(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }

  /* Even rows, even columns */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    if((nx-1) % 2 == 0)
      starsolve_rm(nx, ny, incx, incy,
		   xfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, j);
    
    for(i=2; i<nx-1; i+=2)
      starsolve(nx, ny, incx, incy,
		axy, mx, my, eps,
		bx, by, xx, xy, i, j);

    starsolve_lm(nx, ny, incx, incy,
		 xfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy, j);
  }

  /* Bottom line, even columns */
#pragma omp single
  {
    if((nx-1) % 2 == 0)
      starsolve_rb(nx, ny, incx, incy,
		   xfactor, yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy);

    for(i=2; i<nx-1; i+=2)
      starsolve_mb(nx, ny, incx, incy,
		   yfactor,
		   axy, mx, my, eps,
		   bx, by, xx, xy, i);

    starsolve_lb(nx, ny, incx, incy,
		 xfactor, yfactor,
		 axy, mx, my, eps,
		 bx, by, xx, xy);
  }
}
#endif

#ifdef SMOOTHER_INTERLEAVED
void
gsforward_simple_edge2d(field alpha, field beta,
			const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  field x0, x1, x2, y00, y01, y10, y11;
  field y0, y1, y2, x00, x01, x10, x11;
  field defect, diagonal;
  int i, j;

  assert(x->gr == b->gr);

  /* Bottom row */
#pragma omp single
  {
    x0 = iyfactor * xx[0+(ny-1)*incx];
    x1 = xx[0];
    x2 = xx[0+incx];
    y00 = iyfactor * xy[0+(ny-1)*incy];
    y01 = xy[0];
    y10 = iyfactor * xy[1+(ny-1)*incy];
    y11 = xy[1];
    
    defect = (bx[0]
	      - axy * eps[0+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-1)*nx] + axy * eps[0] + mx * 4.0;
    xx[0] += defect / diagonal;
    
    y0 = ixfactor * xy[nx-1];
    y1 = xy[0];
    y2 = xy[1];
    x00 = ixfactor * xx[nx-1];
    x10 = xx[0];
    x01 = ixfactor * xx[(nx-1)+incx];
    x11 = xx[0+incx];
    
    defect = (by[0]
	      - axy * eps[nx-1] * (y1 - y0 + x00 - x01)
	      - axy * eps[0] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-1] + axy * eps[0] + my * 4.0;
    xy[0] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      x0 = iyfactor * xx[i+(ny-1)*incx];
      x1 = xx[i];
      x2 = xx[i+incx];
      y00 = iyfactor * xy[i+(ny-1)*incy];
      y01 = xy[i];
      y10 = iyfactor * xy[(i+1)+(ny-1)*incy];
      y11 = xy[i+1];

      defect = (bx[i]
		- axy * eps[i+(ny-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-1)*nx] + axy * eps[i] + mx * 4.0;
      xx[i] += defect / diagonal;

      y0 = xy[i-1];
      y1 = xy[i];
      y2 = xy[i+1];
      x00 = xx[i-1];
      x10 = xx[i];
      x01 = xx[(i-1)+incx];
      x11 = xx[i+incx];

      defect = (by[i]
		- axy * eps[i-1] * (y1 - y0 + x00 - x01)
		- axy * eps[i] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[i-1] + axy * eps[i] + my * 4.0;
      xy[i] += defect / diagonal;
    }

    x0 = iyfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = xx[nx-1];
    x2 = xx[(nx-1)+incx];
    y00 = iyfactor * xy[(nx-1)+(ny-1)*incy];
    y01 = xy[nx-1];
    y10 = xfactor * iyfactor * xy[0+(ny-1)*incy];
    y11 = xfactor * xy[0];

    defect = (bx[(nx-1)]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[nx-1] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[nx-1] + mx * 4.0;
    xx[nx-1] += defect / diagonal;
    
    y0 = xy[nx-2];
    y1 = xy[nx-1];
    y2 = xfactor * xy[0];
    x00 = xx[nx-2];
    x10 = xx[nx-1];
    x01 = xx[(nx-2)+incx];
    x11 = xx[(nx-1)+incx];
    
    defect = (by[nx-1]
	      - axy * eps[nx-2] * (y1 - y0 + x00 - x01)
	      - axy * eps[nx-1] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-2] + axy * eps[nx-1] + my * 4.0;
    xy[nx-1] += defect / diagonal;
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
    
    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;

    for(i=1; i<nx-1; i++) {
      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;

      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;
    }

    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;

    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;

    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;

    for(i=1; i<nx-1; i++) {
      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;

      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;
    }

    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
    
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
  }

  /* Top row */
#pragma omp single
  {
    x0 = xx[0+(ny-2)*incx];
    x1 = xx[0+(ny-1)*incx];
    x2 = yfactor * xx[0];
    y00 = xy[0+(ny-2)*incy];
    y01 = xy[0+(ny-1)*incy];
    y10 = xy[1+(ny-2)*incy];
    y11 = xy[1+(ny-1)*incy];
    
    defect = (bx[0+(ny-1)*incx]
	      - axy * eps[0+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-2)*nx] + axy * eps[0+(ny-1)*nx] + mx * 4.0;
    xx[0+(ny-1)*incx] += defect / diagonal;

    y0 = ixfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xy[0+(ny-1)*incy];
    y2 = xy[1+(ny-1)*incy];
    x00 = ixfactor * xx[(nx-1)+(ny-1)*incx];
    x10 = xx[0+(ny-1)*incx];
    x01 = ixfactor * yfactor * xx[nx-1];
    x11 = yfactor * xx[0];
    
    defect = (by[0+(ny-1)*incy]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[0+(ny-1)*nx] + my * 4.0;
    xy[0+(ny-1)*incy] += defect / diagonal;

    for(i=1; i<nx-1; i++) {
      x0 = xx[i+(ny-2)*incx];
      x1 = xx[i+(ny-1)*incx];
      x2 = yfactor * xx[i];
      y00 = xy[i+(ny-2)*incy];
      y01 = xy[i+(ny-1)*incy];
      y10 = xy[(i+1)+(ny-2)*incy];
      y11 = xy[(i+1)+(ny-1)*incy];

      defect = (bx[i+(ny-1)*incx]
		- axy * eps[i+(ny-2)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+(ny-1)*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-2)*nx] + axy * eps[i+(ny-1)*nx] + mx * 4.0;
      xx[i+(ny-1)*incx] += defect / diagonal;

      y0 = xy[(i-1)+(ny-1)*incy];
      y1 = xy[i+(ny-1)*incy];
      y2 = xy[(i+1)+(ny-1)*incy];
      x00 = xx[(i-1)+(ny-1)*incx];
      x10 = xx[i+(ny-1)*incx];
      x01 = yfactor * xx[i-1];
      x11 = yfactor * xx[i];

      defect = (by[i+(ny-1)*incy]
		- axy * eps[(i-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+(ny-1)*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+(ny-1)*nx] + axy * eps[i+(ny-1)*nx] + my * 4.0;
      xy[i+(ny-1)*incy] += defect / diagonal;
    }

    x0 = xx[(nx-1)+(ny-2)*incx];
    x1 = xx[(nx-1)+(ny-1)*incx];
    x2 = yfactor * xx[nx-1];
    y00 = xy[(nx-1)+(ny-2)*incy];
    y01 = xy[(nx-1)+(ny-1)*incy];
    y10 = xfactor * xy[0+(ny-2)*incy];
    y11 = xfactor * xy[0+(ny-1)*incy];

    defect = (bx[(nx-1)+(ny-1)*incx]
	      - axy * eps[(nx-1)+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-2)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + mx * 4.0;
    xx[(nx-1)+(ny-1)*incx] += defect / diagonal;

    y0 = xy[(nx-2)+(ny-1)*incy];
    y1 = xy[(nx-1)+(ny-1)*incy];
    y2 = xfactor * xy[0+(ny-1)*incy];
    x00 = xx[(nx-2)+(ny-1)*incx];
    x10 = xx[(nx-1)+(ny-1)*incx];
    x01 = yfactor * xx[nx-2];
    x11 = yfactor * xx[nx-1];
    
    defect = (by[(nx-1)+(ny-1)*incy]
	      - axy * eps[(nx-2)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+(ny-1)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + my * 4.0;
    xy[(nx-1)+(ny-1)*incy] += defect / diagonal;
  }
}
#else
void
gsforward_simple_edge2d(field alpha, field beta,
			const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  field x0, x1, x2, y00, y01, y10, y11;
  field y0, y1, y2, x00, x01, x10, x11;
  field defect, diagonal;
  int i, j;

  assert(x->gr == b->gr);

  /* Bottom row */
#pragma omp single
  {
    x0 = iyfactor * xx[0+(ny-1)*incx];
    x1 = xx[0];
    x2 = xx[0+incx];
    y00 = iyfactor * xy[0+(ny-1)*incy];
    y01 = xy[0];
    y10 = iyfactor * xy[1+(ny-1)*incy];
    y11 = xy[1];
    
    defect = (bx[0]
	      - axy * eps[0+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-1)*nx] + axy * eps[0] + mx * 4.0;
    xx[0] += defect / diagonal;
    

    for(i=1; i<nx-1; i++) {
      x0 = iyfactor * xx[i+(ny-1)*incx];
      x1 = xx[i];
      x2 = xx[i+incx];
      y00 = iyfactor * xy[i+(ny-1)*incy];
      y01 = xy[i];
      y10 = iyfactor * xy[(i+1)+(ny-1)*incy];
      y11 = xy[i+1];

      defect = (bx[i]
		- axy * eps[i+(ny-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-1)*nx] + axy * eps[i] + mx * 4.0;
      xx[i] += defect / diagonal;
    }

    x0 = iyfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = xx[nx-1];
    x2 = xx[(nx-1)+incx];
    y00 = iyfactor * xy[(nx-1)+(ny-1)*incy];
    y01 = xy[nx-1];
    y10 = xfactor * iyfactor * xy[0+(ny-1)*incy];
    y11 = xfactor * xy[0];

    defect = (bx[(nx-1)]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[nx-1] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[nx-1] + mx * 4.0;
    xx[nx-1] += defect / diagonal;
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;
    }

    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;
    }

    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
  }

  /* Top row */
#pragma omp single
  {
    x0 = xx[0+(ny-2)*incx];
    x1 = xx[0+(ny-1)*incx];
    x2 = yfactor * xx[0];
    y00 = xy[0+(ny-2)*incy];
    y01 = xy[0+(ny-1)*incy];
    y10 = xy[1+(ny-2)*incy];
    y11 = xy[1+(ny-1)*incy];
    
    defect = (bx[0+(ny-1)*incx]
	      - axy * eps[0+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-2)*nx] + axy * eps[0+(ny-1)*nx] + mx * 4.0;
    xx[0+(ny-1)*incx] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      x0 = xx[i+(ny-2)*incx];
      x1 = xx[i+(ny-1)*incx];
      x2 = yfactor * xx[i];
      y00 = xy[i+(ny-2)*incy];
      y01 = xy[i+(ny-1)*incy];
      y10 = xy[(i+1)+(ny-2)*incy];
      y11 = xy[(i+1)+(ny-1)*incy];

      defect = (bx[i+(ny-1)*incx]
		- axy * eps[i+(ny-2)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+(ny-1)*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-2)*nx] + axy * eps[i+(ny-1)*nx] + mx * 4.0;
      xx[i+(ny-1)*incx] += defect / diagonal;
    }

    x0 = xx[(nx-1)+(ny-2)*incx];
    x1 = xx[(nx-1)+(ny-1)*incx];
    x2 = yfactor * xx[nx-1];
    y00 = xy[(nx-1)+(ny-2)*incy];
    y01 = xy[(nx-1)+(ny-1)*incy];
    y10 = xfactor * xy[0+(ny-2)*incy];
    y11 = xfactor * xy[0+(ny-1)*incy];

    defect = (bx[(nx-1)+(ny-1)*incx]
	      - axy * eps[(nx-1)+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-2)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + mx * 4.0;
    xx[(nx-1)+(ny-1)*incx] += defect / diagonal;
  }

  /* Bottom row */
#pragma omp single
  {
    y0 = ixfactor * xy[nx-1];
    y1 = xy[0];
    y2 = xy[1];
    x00 = ixfactor * xx[nx-1];
    x10 = xx[0];
    x01 = ixfactor * xx[(nx-1)+incx];
    x11 = xx[0+incx];
    
    defect = (by[0]
	      - axy * eps[nx-1] * (y1 - y0 + x00 - x01)
	      - axy * eps[0] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-1] + axy * eps[0] + my * 4.0;
    xy[0] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      y0 = xy[i-1];
      y1 = xy[i];
      y2 = xy[i+1];
      x00 = xx[i-1];
      x10 = xx[i];
      x01 = xx[(i-1)+incx];
      x11 = xx[i+incx];

      defect = (by[i]
		- axy * eps[i-1] * (y1 - y0 + x00 - x01)
		- axy * eps[i] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[i-1] + axy * eps[i] + my * 4.0;
      xy[i] += defect / diagonal;
    }

    y0 = xy[nx-2];
    y1 = xy[nx-1];
    y2 = xfactor * xy[0];
    x00 = xx[nx-2];
    x10 = xx[nx-1];
    x01 = xx[(nx-2)+incx];
    x11 = xx[(nx-1)+incx];
    
    defect = (by[nx-1]
	      - axy * eps[nx-2] * (y1 - y0 + x00 - x01)
	      - axy * eps[nx-1] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-2] + axy * eps[nx-1] + my * 4.0;
    xy[nx-1] += defect / diagonal;
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;
    }

    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;
    }

    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
  }

  /* Top row */
#pragma omp single
  {
    y0 = ixfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xy[0+(ny-1)*incy];
    y2 = xy[1+(ny-1)*incy];
    x00 = ixfactor * xx[(nx-1)+(ny-1)*incx];
    x10 = xx[0+(ny-1)*incx];
    x01 = ixfactor * yfactor * xx[nx-1];
    x11 = yfactor * xx[0];
    
    defect = (by[0+(ny-1)*incy]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[0+(ny-1)*nx] + my * 4.0;
    xy[0+(ny-1)*incy] += defect / diagonal;
    
    for(i=1; i<nx-1; i++) {
      y0 = xy[(i-1)+(ny-1)*incy];
      y1 = xy[i+(ny-1)*incy];
      y2 = xy[(i+1)+(ny-1)*incy];
      x00 = xx[(i-1)+(ny-1)*incx];
      x10 = xx[i+(ny-1)*incx];
      x01 = yfactor * xx[i-1];
      x11 = yfactor * xx[i];

      defect = (by[i+(ny-1)*incy]
		- axy * eps[(i-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+(ny-1)*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+(ny-1)*nx] + axy * eps[i+(ny-1)*nx] + my * 4.0;
      xy[i+(ny-1)*incy] += defect / diagonal;
    }

    y0 = xy[(nx-2)+(ny-1)*incy];
    y1 = xy[(nx-1)+(ny-1)*incy];
    y2 = xfactor * xy[0+(ny-1)*incy];
    x00 = xx[(nx-2)+(ny-1)*incx];
    x10 = xx[(nx-1)+(ny-1)*incx];
    x01 = yfactor * xx[nx-2];
    x11 = yfactor * xx[nx-1];
    
    defect = (by[(nx-1)+(ny-1)*incy]
	      - axy * eps[(nx-2)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+(ny-1)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + my * 4.0;
    xy[(nx-1)+(ny-1)*incy] += defect / diagonal;
  }
}
#endif

#ifdef SMOOTHER_INTERLEAVED
void
gsbackward_simple_edge2d(field alpha, field beta,
			 const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  field x0, x1, x2, y00, y01, y10, y11;
  field y0, y1, y2, x00, x01, x10, x11;
  field defect, diagonal;
  int i, j;

  assert(x->gr == b->gr);

  /* Top row */
#pragma omp single
  {
    y0 = xy[(nx-2)+(ny-1)*incy];
    y1 = xy[(nx-1)+(ny-1)*incy];
    y2 = xfactor * xy[0+(ny-1)*incy];
    x00 = xx[(nx-2)+(ny-1)*incx];
    x10 = xx[(nx-1)+(ny-1)*incx];
    x01 = yfactor * xx[nx-2];
    x11 = yfactor * xx[nx-1];
    
    defect = (by[(nx-1)+(ny-1)*incy]
	      - axy * eps[(nx-2)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+(ny-1)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + my * 4.0;
    xy[(nx-1)+(ny-1)*incy] += defect / diagonal;
    
    x0 = xx[(nx-1)+(ny-2)*incx];
    x1 = xx[(nx-1)+(ny-1)*incx];
    x2 = yfactor * xx[nx-1];
    y00 = xy[(nx-1)+(ny-2)*incy];
    y01 = xy[(nx-1)+(ny-1)*incy];
    y10 = xfactor * xy[0+(ny-2)*incy];
    y11 = xfactor * xy[0+(ny-1)*incy];

    defect = (bx[(nx-1)+(ny-1)*incx]
	      - axy * eps[(nx-1)+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-2)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + mx * 4.0;
    xx[(nx-1)+(ny-1)*incx] += defect / diagonal;
    
    for(i=nx-2; i>0; i--) {
      y0 = xy[(i-1)+(ny-1)*incy];
      y1 = xy[i+(ny-1)*incy];
      y2 = xy[(i+1)+(ny-1)*incy];
      x00 = xx[(i-1)+(ny-1)*incx];
      x10 = xx[i+(ny-1)*incx];
      x01 = yfactor * xx[i-1];
      x11 = yfactor * xx[i];

      defect = (by[i+(ny-1)*incy]
		- axy * eps[(i-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+(ny-1)*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+(ny-1)*nx] + axy * eps[i+(ny-1)*nx] + my * 4.0;
      xy[i+(ny-1)*incy] += defect / diagonal;

      x0 = xx[i+(ny-2)*incx];
      x1 = xx[i+(ny-1)*incx];
      x2 = yfactor * xx[i];
      y00 = xy[i+(ny-2)*incy];
      y01 = xy[i+(ny-1)*incy];
      y10 = xy[(i+1)+(ny-2)*incy];
      y11 = xy[(i+1)+(ny-1)*incy];

      defect = (bx[i+(ny-1)*incx]
		- axy * eps[i+(ny-2)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+(ny-1)*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-2)*nx] + axy * eps[i+(ny-1)*nx] + mx * 4.0;
      xx[i+(ny-1)*incx] += defect / diagonal;
    }

    y0 = ixfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xy[0+(ny-1)*incy];
    y2 = xy[1+(ny-1)*incy];
    x00 = ixfactor * xx[(nx-1)+(ny-1)*incx];
    x10 = xx[0+(ny-1)*incx];
    x01 = ixfactor * yfactor * xx[nx-1];
    x11 = yfactor * xx[0];
    
    defect = (by[0+(ny-1)*incy]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[0+(ny-1)*nx] + my * 4.0;
    xy[0+(ny-1)*incy] += defect / diagonal;

    x0 = xx[0+(ny-2)*incx];
    x1 = xx[0+(ny-1)*incx];
    x2 = yfactor * xx[0];
    y00 = xy[0+(ny-2)*incy];
    y01 = xy[0+(ny-1)*incy];
    y10 = xy[1+(ny-2)*incy];
    y11 = xy[1+(ny-1)*incy];
    
    defect = (bx[0+(ny-1)*incx]
	      - axy * eps[0+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-2)*nx] + axy * eps[0+(ny-1)*nx] + mx * 4.0;
    xx[0+(ny-1)*incx] += defect / diagonal;
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
    
    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
    
    for(i=nx-2; i>0; i--) {
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;

      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;
    }

    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;

    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
    
    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
    
    for(i=nx-2; i>0; i--) {
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;

      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;
    }

    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;

    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
  }

  /* Bottom row */
#pragma omp single
  {
    y0 = xy[nx-2];
    y1 = xy[nx-1];
    y2 = xfactor * xy[0];
    x00 = xx[nx-2];
    x10 = xx[nx-1];
    x01 = xx[(nx-2)+incx];
    x11 = xx[(nx-1)+incx];
    
    defect = (by[nx-1]
	      - axy * eps[nx-2] * (y1 - y0 + x00 - x01)
	      - axy * eps[nx-1] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-2] + axy * eps[nx-1] + my * 4.0;
    xy[nx-1] += defect / diagonal;
    
    x0 = iyfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = xx[nx-1];
    x2 = xx[(nx-1)+incx];
    y00 = iyfactor * xy[(nx-1)+(ny-1)*incy];
    y01 = xy[nx-1];
    y10 = xfactor * iyfactor * xy[0+(ny-1)*incy];
    y11 = xfactor * xy[0];

    defect = (bx[(nx-1)]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[nx-1] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[nx-1] + mx * 4.0;
    xx[nx-1] += defect / diagonal;

    for(i=nx-2; i>0; i--) {
      y0 = xy[i-1];
      y1 = xy[i];
      y2 = xy[i+1];
      x00 = xx[i-1];
      x10 = xx[i];
      x01 = xx[(i-1)+incx];
      x11 = xx[i+incx];

      defect = (by[i]
		- axy * eps[i-1] * (y1 - y0 + x00 - x01)
		- axy * eps[i] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[i-1] + axy * eps[i] + my * 4.0;
      xy[i] += defect / diagonal;

      x0 = iyfactor * xx[i+(ny-1)*incx];
      x1 = xx[i];
      x2 = xx[i+incx];
      y00 = iyfactor * xy[i+(ny-1)*incy];
      y01 = xy[i];
      y10 = iyfactor * xy[(i+1)+(ny-1)*incy];
      y11 = xy[i+1];

      defect = (bx[i]
		- axy * eps[i+(ny-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-1)*nx] + axy * eps[i] + mx * 4.0;
      xx[i] += defect / diagonal;
    }

    y0 = ixfactor * xy[nx-1];
    y1 = xy[0];
    y2 = xy[1];
    x00 = ixfactor * xx[nx-1];
    x10 = xx[0];
    x01 = ixfactor * xx[(nx-1)+incx];
    x11 = xx[0+incx];
    
    defect = (by[0]
	      - axy * eps[nx-1] * (y1 - y0 + x00 - x01)
	      - axy * eps[0] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-1] + axy * eps[0] + my * 4.0;
    xy[0] += defect / diagonal;

    x0 = iyfactor * xx[0+(ny-1)*incx];
    x1 = xx[0];
    x2 = xx[0+incx];
    y00 = iyfactor * xy[0+(ny-1)*incy];
    y01 = xy[0];
    y10 = iyfactor * xy[1+(ny-1)*incy];
    y11 = xy[1];
    
    defect = (bx[0]
	      - axy * eps[0+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-1)*nx] + axy * eps[0] + mx * 4.0;
    xx[0] += defect / diagonal;
  }
}
#else
void
gsbackward_simple_edge2d(field alpha, field beta,
			 const edge2d *b, edge2d *x)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *bx = b->x + incx + 1;
  const field *by = b->y + incy + 1;
  field *xx = x->x + incx + 1;
  field *xy = x->y + incy + 1;
  field xfactor = x->gr->xfactor;
  field yfactor = x->gr->yfactor;
  field ixfactor = CONJ(xfactor);
  field iyfactor = CONJ(yfactor);
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = x->gr->eps;
  field x0, x1, x2, y00, y01, y10, y11;
  field y0, y1, y2, x00, x01, x10, x11;
  field defect, diagonal;
  int i, j;

  assert(x->gr == b->gr);

  /* Top row */
#pragma omp single
  {
    y0 = xy[(nx-2)+(ny-1)*incy];
    y1 = xy[(nx-1)+(ny-1)*incy];
    y2 = xfactor * xy[0+(ny-1)*incy];
    x00 = xx[(nx-2)+(ny-1)*incx];
    x10 = xx[(nx-1)+(ny-1)*incx];
    x01 = yfactor * xx[nx-2];
    x11 = yfactor * xx[nx-1];
    
    defect = (by[(nx-1)+(ny-1)*incy]
	      - axy * eps[(nx-2)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+(ny-1)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + my * 4.0;
    xy[(nx-1)+(ny-1)*incy] += defect / diagonal;
    
    x0 = xx[(nx-1)+(ny-2)*incx];
    x1 = xx[(nx-1)+(ny-1)*incx];
    x2 = yfactor * xx[nx-1];
    y00 = xy[(nx-1)+(ny-2)*incy];
    y01 = xy[(nx-1)+(ny-1)*incy];
    y10 = xfactor * xy[0+(ny-2)*incy];
    y11 = xfactor * xy[0+(ny-1)*incy];

    defect = (bx[(nx-1)+(ny-1)*incx]
	      - axy * eps[(nx-1)+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-2)*nx] + axy * eps[(nx-1)+(ny-1)*nx] + mx * 4.0;
    xx[(nx-1)+(ny-1)*incx] += defect / diagonal;
    
    for(i=nx-2; i>0; i--) {
      y0 = xy[(i-1)+(ny-1)*incy];
      y1 = xy[i+(ny-1)*incy];
      y2 = xy[(i+1)+(ny-1)*incy];
      x00 = xx[(i-1)+(ny-1)*incx];
      x10 = xx[i+(ny-1)*incx];
      x01 = yfactor * xx[i-1];
      x11 = yfactor * xx[i];

      defect = (by[i+(ny-1)*incy]
		- axy * eps[(i-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+(ny-1)*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+(ny-1)*nx] + axy * eps[i+(ny-1)*nx] + my * 4.0;
      xy[i+(ny-1)*incy] += defect / diagonal;

      x0 = xx[i+(ny-2)*incx];
      x1 = xx[i+(ny-1)*incx];
      x2 = yfactor * xx[i];
      y00 = xy[i+(ny-2)*incy];
      y01 = xy[i+(ny-1)*incy];
      y10 = xy[(i+1)+(ny-2)*incy];
      y11 = xy[(i+1)+(ny-1)*incy];

      defect = (bx[i+(ny-1)*incx]
		- axy * eps[i+(ny-2)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+(ny-1)*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-2)*nx] + axy * eps[i+(ny-1)*nx] + mx * 4.0;
      xx[i+(ny-1)*incx] += defect / diagonal;
    }

    y0 = ixfactor * xy[(nx-1)+(ny-1)*incy];
    y1 = xy[0+(ny-1)*incy];
    y2 = xy[1+(ny-1)*incy];
    x00 = ixfactor * xx[(nx-1)+(ny-1)*incx];
    x10 = xx[0+(ny-1)*incx];
    x01 = ixfactor * yfactor * xx[nx-1];
    x11 = yfactor * xx[0];
    
    defect = (by[0+(ny-1)*incy]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+(ny-1)*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[0+(ny-1)*nx] + my * 4.0;
    xy[0+(ny-1)*incy] += defect / diagonal;

    x0 = xx[0+(ny-2)*incx];
    x1 = xx[0+(ny-1)*incx];
    x2 = yfactor * xx[0];
    y00 = xy[0+(ny-2)*incy];
    y01 = xy[0+(ny-1)*incy];
    y10 = xy[1+(ny-2)*incy];
    y11 = xy[1+(ny-1)*incy];
    
    defect = (bx[0+(ny-1)*incx]
	      - axy * eps[0+(ny-2)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+(ny-1)*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-2)*nx] + axy * eps[0+(ny-1)*nx] + mx * 4.0;
    xx[0+(ny-1)*incx] += defect / diagonal;
  }

  /* Odd rows */
#pragma omp for
  for(j=1; j<ny-1; j+=2) {
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
    
    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
    
    for(i=nx-2; i>0; i--) {
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;

      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;
    }

    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;

    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
  }

  /* Even rows */
#pragma omp for
  for(j=2; j<ny-1; j+=2) {
    y0 = xy[(nx-2)+j*incy];
    y1 = xy[(nx-1)+j*incy];
    y2 = xfactor * xy[0+j*incy];
    x00 = xx[(nx-2)+j*incx];
    x10 = xx[(nx-1)+j*incx];
    x01 = xx[(nx-2)+(j+1)*incx];
    x11 = xx[(nx-1)+(j+1)*incx];
    
    defect = (by[(nx-1)+j*incy]
	      - axy * eps[(nx-2)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[(nx-1)+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-2)+j*nx] + axy * eps[(nx-1)+j*nx] + my * 4.0;
    xy[(nx-1)+j*incy] += defect / diagonal;
    
    x0 = xx[(nx-1)+(j-1)*incx];
    x1 = xx[(nx-1)+j*incx];
    x2 = xx[(nx-1)+(j+1)*incx];
    y00 = xy[(nx-1)+(j-1)*incy];
    y01 = xy[(nx-1)+j*incy];
    y10 = xfactor * xy[0+(j-1)*incy];
    y11 = xfactor * xy[0+j*incy];

    defect = (bx[(nx-1)+j*incx]
	      - axy * eps[(nx-1)+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[(nx-1)+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(j-1)*nx] + axy * eps[(nx-1)+j*nx] + mx * 4.0;
    xx[(nx-1)+j*incx] += defect / diagonal;
    
    for(i=nx-2; i>0; i--) {
      y0 = xy[(i-1)+j*incy];
      y1 = xy[i+j*incy];
      y2 = xy[(i+1)+j*incy];
      x00 = xx[(i-1)+j*incx];
      x10 = xx[i+j*incx];
      x01 = xx[(i-1)+(j+1)*incx];
      x11 = xx[i+(j+1)*incx];

      defect = (by[i+j*incy]
		- axy * eps[(i-1)+j*nx] * (y1 - y0 + x00 - x01)
		- axy * eps[i+j*nx] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[(i-1)+j*nx] + axy * eps[i+j*nx] + my * 4.0;
      xy[i+j*incy] += defect / diagonal;

      x0 = xx[i+(j-1)*incx];
      x1 = xx[i+j*incx];
      x2 = xx[i+(j+1)*incx];
      y00 = xy[i+(j-1)*incy];
      y01 = xy[i+j*incy];
      y10 = xy[(i+1)+(j-1)*incy];
      y11 = xy[(i+1)+j*incy];

      defect = (bx[i+j*incx]
		- axy * eps[i+(j-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i+j*nx] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(j-1)*nx] + axy * eps[i+j*nx] + mx * 4.0;
      xx[i+j*incx] += defect / diagonal;
    }

    y0 = ixfactor * xy[(nx-1)+j*incy];
    y1 = xy[0+j*incy];
    y2 = xy[1+j*incy];
    x00 = ixfactor * xx[(nx-1)+j*incx];
    x10 = xx[0+j*incx];
    x01 = ixfactor * xx[(nx-1)+(j+1)*incx];
    x11 = xx[0+(j+1)*incx];
    
    defect = (by[0+j*incy]
	      - axy * eps[(nx-1)+j*nx] * (y1 - y0 + x00 - x01)
	      - axy * eps[0+j*nx] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[(nx-1)+j*nx] + axy * eps[0+j*nx] + my * 4.0;
    xy[0+j*incy] += defect / diagonal;

    x0 = xx[0+(j-1)*incx];
    x1 = xx[0+j*incx];
    x2 = xx[0+(j+1)*incx];
    y00 = xy[0+(j-1)*incy];
    y01 = xy[0+j*incy];
    y10 = xy[1+(j-1)*incy];
    y11 = xy[1+j*incy];
    
    defect = (bx[0+j*incx]
	      - axy * eps[0+(j-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0+j*nx] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(j-1)*nx] + axy * eps[0+j*nx] + mx * 4.0;
    xx[0+j*incx] += defect / diagonal;
  }

  /* Bottom row */
#pragma omp single
  {
    y0 = xy[nx-2];
    y1 = xy[nx-1];
    y2 = xfactor * xy[0];
    x00 = xx[nx-2];
    x10 = xx[nx-1];
    x01 = xx[(nx-2)+incx];
    x11 = xx[(nx-1)+incx];
    
    defect = (by[nx-1]
	      - axy * eps[nx-2] * (y1 - y0 + x00 - x01)
	      - axy * eps[nx-1] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-2] + axy * eps[nx-1] + my * 4.0;
    xy[nx-1] += defect / diagonal;
    
    x0 = iyfactor * xx[(nx-1)+(ny-1)*incx];
    x1 = xx[nx-1];
    x2 = xx[(nx-1)+incx];
    y00 = iyfactor * xy[(nx-1)+(ny-1)*incy];
    y01 = xy[nx-1];
    y10 = xfactor * iyfactor * xy[0+(ny-1)*incy];
    y11 = xfactor * xy[0];

    defect = (bx[(nx-1)]
	      - axy * eps[(nx-1)+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[nx-1] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[(nx-1)+(ny-1)*nx] + axy * eps[nx-1] + mx * 4.0;
    xx[nx-1] += defect / diagonal;

    for(i=nx-2; i>0; i--) {
      y0 = xy[i-1];
      y1 = xy[i];
      y2 = xy[i+1];
      x00 = xx[i-1];
      x10 = xx[i];
      x01 = xx[(i-1)+incx];
      x11 = xx[i+incx];

      defect = (by[i]
		- axy * eps[i-1] * (y1 - y0 + x00 - x01)
		- axy * eps[i] * (y1 - y2 + x11 - x10)
		- my * (4.0 * y1 + y0 + y2));
      diagonal = axy * eps[i-1] + axy * eps[i] + my * 4.0;
      xy[i] += defect / diagonal;

      x0 = iyfactor * xx[i+(ny-1)*incx];
      x1 = xx[i];
      x2 = xx[i+incx];
      y00 = iyfactor * xy[i+(ny-1)*incy];
      y01 = xy[i];
      y10 = iyfactor * xy[(i+1)+(ny-1)*incy];
      y11 = xy[i+1];

      defect = (bx[i]
		- axy * eps[i+(ny-1)*nx] * (x1 - x0 + y00 - y10)
		- axy * eps[i] * (x1 - x2 + y11 - y01)
		- mx * (4.0 * x1 + x0 + x2));
      diagonal = axy * eps[i+(ny-1)*nx] + axy * eps[i] + mx * 4.0;
      xx[i] += defect / diagonal;
    }

    y0 = ixfactor * xy[nx-1];
    y1 = xy[0];
    y2 = xy[1];
    x00 = ixfactor * xx[nx-1];
    x10 = xx[0];
    x01 = ixfactor * xx[(nx-1)+incx];
    x11 = xx[0+incx];
    
    defect = (by[0]
	      - axy * eps[nx-1] * (y1 - y0 + x00 - x01)
	      - axy * eps[0] * (y1 - y2 + x11 - x10)
	      - my * (4.0 * y1 + y0 + y2));
    diagonal = axy * eps[nx-1] + axy * eps[0] + my * 4.0;
    xy[0] += defect / diagonal;

    x0 = iyfactor * xx[0+(ny-1)*incx];
    x1 = xx[0];
    x2 = xx[0+incx];
    y00 = iyfactor * xy[0+(ny-1)*incy];
    y01 = xy[0];
    y10 = iyfactor * xy[1+(ny-1)*incy];
    y11 = xy[1];
    
    defect = (bx[0]
	      - axy * eps[0+(ny-1)*nx] * (x1 - x0 + y00 - y10)
	      - axy * eps[0] * (x1 - x2 + y11 - y01)
	      - mx * (4.0 * x1 + x0 + x2));
    diagonal = axy * eps[0+(ny-1)*nx] + axy * eps[0] + mx * 4.0;
    xx[0] += defect / diagonal;
  }
}
#endif

void
gssymm_simple_edge2d(field alpha, field beta,
		     const edge2d *b, edge2d *x)
{
  gsforward_simple_edge2d(alpha, beta, b, x);
  gsbackward_simple_edge2d(alpha, beta, b, x);
}

void
prolongation_edge2d(field alpha, const edge2d *c, edge2d *f)
{
  int cnx = c->gr->nx;
  int cny = c->gr->ny;
  int fnx = f->gr->nx;
  int cincx = cnx + 1;
  int cincy = cnx + 2;
  int fincx = fnx + 1;
  int fincy = fnx + 2;
  const field *cx = c->x + cincx + 1;
  const field *cy = c->y + cincy + 1;
  field *fx = f->x + fincx + 1;
  field *fy = f->y + fincy + 1;
  field xfactor = f->gr->xfactor;
  field yfactor = f->gr->yfactor;
  int i, j;

  assert(f->gr->nx == 2 * c->gr->nx);
  assert(f->gr->ny == 2 * c->gr->ny);

  /* Interpolate x edges */
#pragma omp for
  for(j=0; j<cny-1; j++)
    for(i=0; i<cnx; i++) {
      fx[2*i  + 2*j   *fincx] += alpha * 0.5 * cx[i+j*cincx];
      fx[2*i+1+ 2*j   *fincx] += alpha * 0.5 * cx[i+j*cincx];
      fx[2*i  +(2*j+1)*fincx] += alpha * 0.25 * (cx[i+j*cincx] + cx[i+(j+1)*cincx]);
      fx[2*i+1+(2*j+1)*fincx] += alpha * 0.25 * (cx[i+j*cincx] + cx[i+(j+1)*cincx]);
    }

  /* Top line wraps around to bottom */
#pragma omp single
  {
    j = cny-1;
    for(i=0; i<cnx; i++) {
      fx[2*i  + 2*j   *fincx] += alpha * 0.5 * cx[i+j*cincx];
      fx[2*i+1+ 2*j   *fincx] += alpha * 0.5 * cx[i+j*cincx];
      fx[2*i  +(2*j+1)*fincx] += alpha * 0.25 * (cx[i+j*cincx] + yfactor * cx[i]);
      fx[2*i+1+(2*j+1)*fincx] += alpha * 0.25 * (cx[i+j*cincx] + yfactor * cx[i]);
    }
  }
  
  /* Interpolate y edges */
#pragma omp for
  for(j=0; j<cny; j++) {
    for(i=0; i<cnx-1; i++) {
      fy[ 2*i   + 2*j   *fincy] += alpha * 0.5 * cy[i+j*cincy];
      fy[ 2*i   +(2*j+1)*fincy] += alpha * 0.5 * cy[i+j*cincy];
      fy[(2*i+1)+ 2*j   *fincy] += alpha * 0.25 * (cy[i+j*cincy] + cy[(i+1)+j*cincy]);
      fy[(2*i+1)+(2*j+1)*fincy] += alpha * 0.25 * (cy[i+j*cincy] + cy[(i+1)+j*cincy]);
    }

    /* Right line wraps around to left */
    i = cnx-1;
    fy[ 2*i   + 2*j   *fincy] += alpha * 0.5 * cy[i+j*cincy];
    fy[ 2*i   +(2*j+1)*fincy] += alpha * 0.5 * cy[i+j*cincy];
    fy[(2*i+1)+ 2*j   *fincy] += alpha * 0.25 * (cy[i+j*cincy] + xfactor * cy[j*cincy]);
    fy[(2*i+1)+(2*j+1)*fincy] += alpha * 0.25 * (cy[i+j*cincy] + xfactor * cy[j*cincy]);
  }
}

void
restriction_edge2d(field alpha, const edge2d *f, edge2d *c)
{
  int cnx = c->gr->nx;
  int cny = c->gr->ny;
  int fnx = f->gr->nx;
  int fny = f->gr->ny;
  int cincx = cnx + 1;
  int cincy = cnx + 2;
  int fincx = fnx + 1;
  int fincy = fnx + 2;
  field *cx = c->x + cincx + 1;
  field *cy = c->y + cincy + 1;
  const field *fx = f->x + fincx + 1;
  const field *fy = f->y + fincy + 1;
  field xfactor = f->gr->xfactor;
  field yfactor = f->gr->yfactor;
  int i, j;
  
  assert(f->gr->nx == 2 * c->gr->nx);
  assert(f->gr->ny == 2 * c->gr->ny);
  
  /* Accumulate x edges */
#pragma omp single
  for(i=0; i<cnx; i++)
    cx[i] += alpha * (0.5 *    fx[2*i      ]
		      + 0.5  * fx[2*i+1    ]
		      + 0.25 * fx[2*i  +(fny-1)*fincx] * CONJ(yfactor)
		      + 0.25 * fx[2*i+1+(fny-1)*fincx] * CONJ(yfactor)
		      + 0.25 * fx[2*i  +fincx]
		      + 0.25 * fx[2*i+1+fincx]);

#pragma omp for
  for(j=1; j<cny; j++)
    for(i=0; i<cnx; i++)
      cx[i+j*cincx] += alpha * (0.5 *    fx[ 2*i   + 2*j   *fincx]
				+ 0.5  * fx[(2*i+1)+ 2*j   *fincx]
				+ 0.25 * fx[ 2*i   +(2*j-1)*fincx]
				+ 0.25 * fx[(2*i+1)+(2*j-1)*fincx]
				+ 0.25 * fx[ 2*i   +(2*j+1)*fincx]
				+ 0.25 * fx[(2*i+1)+(2*j+1)*fincx]);

  /* Accumulate y edges */
#pragma omp for
  for(j=0; j<cny; j++) {
    cy[j*cincy] += alpha * (0.5  * fy[           2*j   *fincy]
			    + 0.5  * fy[        (2*j+1)*fincy]
			    + 0.25 * fy[(fnx-1)+ 2*j   *fincy] * CONJ(xfactor)
			    + 0.25 * fy[(fnx-1)+(2*j+1)*fincy] * CONJ(xfactor)
			    + 0.25 * fy[1      + 2*j   *fincy]
			    + 0.25 * fy[1      +(2*j+1)*fincy]);
    
    for(i=1; i<cnx; i++)
      cy[i+j*cincy] += alpha * (0.5  * fy[2*i      + 2*j   *fincy]
				+ 0.5  * fy[2*i    +(2*j+1)*fincy]
				+ 0.25 * fy[(2*i+1)+ 2*j   *fincy]
				+ 0.25 * fy[(2*i+1)+(2*j+1)*fincy]
				+ 0.25 * fy[(2*i-1)+ 2*j   *fincy]
				+ 0.25 * fy[(2*i-1)+(2*j+1)*fincy]);
  }
}

void
vcycle_edge2d(int L, int nu, const matrix *Ac,
	      edge2d **b, edge2d **x, edge2d **d)
{
  field *xd = 0;
  int i, l;
  
  for(l=L; l>0; l--) {
    for(i=0; i<nu; i++)
      gsforward_edge2d(1.0, 1.0, b[l], x[l]);
    
    copy_edge2d(b[l], d[l]);
    addeval_edge2d(-1.0, -1.0, x[l], d[l]);
    
    zero_edge2d(b[l-1]);
    restriction_edge2d(1.0, d[l], b[l-1]);
    
    zero_edge2d(x[l-1]);
  }

  if(Ac) {
    assert(Ac->rows == 2 * x[0]->gr->nx * x[0]->gr->ny);

#pragma omp single
    {
      xd = (field *) malloc(sizeof(field) * Ac->rows);
    
      densefrom_edge2d(b[0], xd);

      potrsv_matrix(Ac, xd);
      
      denseto_edge2d(xd, x[0]);

      free(xd);
    }
  }
  else {
    for(i=0; i<nu; i++)
      gssymm_edge2d(1.0, 1.0, b[0], x[0]);
  }
  
  for(l=1; l<=L; l++) {
    prolongation_edge2d(1.0, x[l-1], x[l]);

    for(i=0; i<nu; i++)
      gsbackward_edge2d(1.0, 1.0, b[l], x[l]);
  }
}

void
hcycle_edge2d(int L, int nu, const matrix *Ac,
	      edge2d **b, edge2d **x, edge2d **d,
	      node2d **bg, node2d **xg)
{
  field *xd = 0;
  int i, l;
  
  for(l=L; l>0; l--) {
    for(i=0; i<nu; i++)
      gsforward_simple_edge2d(1.0, 1.0, b[l], x[l]);
    
    copy_edge2d(b[l], d[l]);
    addeval_edge2d(-1.0, -1.0, x[l], d[l]);

    zero_node2d(bg[l]);
    adjgradient_node2d(1.0, d[l], bg[l]);
    zero_node2d(xg[l]);
    for(i=0; i<nu; i++)
      gsforward_node2d(1.0, bg[l], xg[l]);
    gradient_node2d(1.0, xg[l], x[l]);
    
    copy_edge2d(b[l], d[l]);
    addeval_edge2d(-1.0, -1.0, x[l], d[l]);

    zero_edge2d(b[l-1]);
    restriction_edge2d(1.0, d[l], b[l-1]);
    
    zero_edge2d(x[l-1]);
  }

  if(Ac) {
    assert(Ac->rows == 2 * x[0]->gr->nx * x[0]->gr->ny);

#pragma omp single
    xd = (field *) malloc(sizeof(field) * Ac->rows);
    
    densefrom_edge2d(b[0], xd);
#pragma omp single
    potrsv_matrix(Ac, xd);

    denseto_edge2d(xd, x[0]);

#pragma omp single
    free(xd);
  }
  else {
    for(i=0; i<nu; i++)
      gssymm_simple_edge2d(1.0, 1.0, b[0], x[0]);
  }
  
  for(l=1; l<=L; l++) {
    prolongation_edge2d(1.0, x[l-1], x[l]);

    copy_edge2d(b[l], d[l]);
    addeval_edge2d(-1.0, -1.0, x[l], d[l]);

    zero_node2d(bg[l]);
    adjgradient_node2d(1.0, d[l], bg[l]);
    zero_node2d(xg[l]);
    for(i=0; i<nu; i++)
      gsbackward_node2d(1.0, bg[l], xg[l]);
    gradient_node2d(1.0, xg[l], x[l]);
    
    for(i=0; i<nu; i++)
      gsbackward_simple_edge2d(1.0, 1.0, b[l], x[l]);
  }
}

void
mgcginit_edge2d(int L, int nu, const edge2d *b, edge2d *x,
		edge2d *p, edge2d *a,
		edge2d **r, edge2d **q, edge2d **d)
{
  copy_edge2d(b, r[L]);
  addeval_edge2d(-1.0, -1.0, x, r[L]);

  zero_edge2d(q[L]);
  vcycle_edge2d(L, nu, 0, r, q, d);

  copy_edge2d(q[L], p);
}

void
mgcgstep_edge2d(int L, int nu, const edge2d *b, edge2d *x,
		edge2d *p, edge2d *a,
		edge2d **r, edge2d **q, edge2d **d)
{
  field gamma, lambda, mu;
  
  zero_edge2d(a);
  addeval_edge2d(1.0, 1.0, p, a);

  gamma = dotprod_edge2d(p, a);
  lambda = dotprod_edge2d(p, r[L]) / gamma;
  
  add_edge2d(lambda, p, x);

#ifdef CG_SAFE_RESIDUAL
  copy_edge2d(b, r[L]);
  addeval_edge2d(-1.0, -1.0, x, r[L]);
#else
  /* Apparently this may lead to poor results */
  add_edge2d(-lambda, a, r[L]);
#endif

  zero_edge2d(q[L]);
  vcycle_edge2d(L, nu, 0, r, q, d);

  mu = dotprod_edge2d(a, q[L]) / gamma;
  scale_edge2d(-mu, p);
  add_edge2d(1.0, q[L], p);
}

void
densematrix_edge2d(field alpha, field beta,
		   const grid2d *gr, matrix *a)
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
  field axy = alpha / hx / hy;
  field mx = beta * hy / hx / 6.0;
  field my = beta * hx / hy / 6.0;
  const real *eps = gr->eps;
  field coeff;
  int off;
  int i, j, ia, ja;

  assert(a->rows == a->cols);
  assert(a->rows == 2 * nx * ny);

  /* Offset to y edges */
  off = nx * ny;

  /* Clear the matrix */
  for(j=0; j<cols; j++)
    for(i=0; i<rows; i++)
      aa[i+lda*j] = 0.0;

  /* Fill rows corresponding to x edges */
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      ia = i + j*nx;

      /* Lower box */
      coeff = (j > 0 ?
	       axy * eps[i+(j-1)*nx] :
	       axy * eps[i+(ny-1)*nx]);
      
      ja = ia;
      aa[ia+ja*lda] += coeff;

      if(j > 0) {
	ja = i + (j-1)*nx + off;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = i + (ny-1)*nx + off;
	aa[ia+ja*lda] += CONJ(yfactor) * coeff;
      }

      if(j > 0) {
	ja = i + (j-1)*nx;
	aa[ia+ja*lda] -= coeff;
      }
      else {
	ja = i + (ny-1)*nx;
	aa[ia+ja*lda] -= CONJ(yfactor) * coeff;
      }

      if(j > 0) {
	if(i < nx-1) {
	  ja = (i+1) + (j-1)*nx + off;
	  aa[ia+ja*lda] -= coeff;
	}
	else {
	  ja = 0 + (j-1)*nx + off;
	  aa[ia+ja*lda] -= xfactor * coeff;
	}
      }
      else {
	if(i < nx - 1) {
	  ja = (i+1) + (ny-1)*nx + off;
	  aa[ia+ja*lda] -= CONJ(yfactor) * coeff;
	}
	else {
	  ja = 0 + (ny-1)*nx + off;
	  aa[ia+ja*lda] -= xfactor * CONJ(yfactor) * coeff;
	}
      }

      /* Upper box */
      coeff = axy * eps[i+j*nx];

      ja = ia;
      aa[ia+ja*lda] += coeff;

      ja = i + j*nx + off;
      aa[ia+ja*lda] -= coeff;

      if(j < ny - 1) {
	ja = i + (j+1)*nx;
	aa[ia+ja*lda] -= coeff;
      }
      else {
	ja = i;
	aa[ia+ja*lda] -= yfactor * coeff;
      }

      if(i < nx - 1) {
	ja = (i+1) + j*nx + off;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = j*nx + off;
	aa[ia+ja*lda] += xfactor * coeff;
      }

      /* Mass matrix */
      coeff = 4.0 * mx;
      ja = ia;
      aa[ia+ja*lda] += coeff;

      coeff = mx;
      if(j > 0) {
	ja = i + (j-1)*nx;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = i + (ny-1)*nx;
	aa[ia+ja*lda] += CONJ(yfactor) * coeff;
      }

      if(j < ny - 1) {
	ja = i + (j+1)*nx;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = i;
	aa[ia+ja*lda] += yfactor * coeff;
      }
    }

  /* Fill rows corresponding to y edges */
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      ia = i + j * nx + off;

      /* Left box */
      coeff = (i > 0 ?
	       axy * eps[(i-1)+j*nx] :
	       axy * eps[(nx-1)+j*nx]);

      ja = ia;
      aa[ia+ja*lda] += coeff;

      if(i > 0) {
	ja = (i-1) + j*nx;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = (nx-1) + j*nx;
	aa[ia+ja*lda] += CONJ(xfactor) * coeff;
      }

      if(i > 0) {
	ja = (i-1) + j*nx + off;
	aa[ia+ja*lda] -= coeff;
      }
      else {
	ja = (nx-1) + j*nx + off;
	aa[ia+ja*lda] -= CONJ(xfactor) * coeff;
      }

      if(i > 0) {
	if(j < ny - 1) {
	  ja = (i-1) + (j+1)*nx;
	  aa[ia+ja*lda] -= coeff;
	}
	else {
	  ja = (i-1);
	  aa[ia+ja*lda] -= yfactor * coeff;
	}
      }
      else {
	if(j < ny - 1) {
	  ja = (nx-1) + (j+1)*nx;
	  aa[ia+ja*lda] -= CONJ(xfactor) * coeff;
	}
	else {
	  ja = (nx-1);
	  aa[ia+ja*lda] -= CONJ(xfactor) * yfactor * coeff;
	}
      }
      
      /* Right box */
      coeff = axy * eps[i+j*nx];

      ja = ia;
      aa[ia+ja*lda] += coeff;

      ja = i + j*nx;
      aa[ia+ja*lda] -= coeff;

      if(i < nx - 1) {
	ja = (i+1) + j*nx + off;
	aa[ia+ja*lda] -= coeff;
      }
      else {
	ja = j*nx + off;
	aa[ia+ja*lda] -= xfactor * coeff;
      }

      if(j < ny - 1) {
	ja = i + (j+1)*nx;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = i;
	aa[ia+ja*lda] += yfactor * coeff;
      }

      /* Mass matrix */
      coeff = 4.0 * my;
      ja = ia;
      aa[ia+ja*lda] += coeff;

      coeff = my;
      if(i > 0) {
	ja = (i-1) + j*nx + off;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = (nx-1) + j*nx + off;
	aa[ia+ja*lda] += CONJ(xfactor) * coeff;
      }

      if(i < nx - 1) {
	ja = (i+1) + j*nx + off;
	aa[ia+ja*lda] += coeff;
      }
      else {
	ja = j*nx + off;
	aa[ia+ja*lda] += xfactor * coeff;
      }
    }
}

void
denseto_edge2d(const field *x, edge2d *y)
{
  int nx = y->gr->nx;
  int ny = y->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  field *yx = y->x + incx + 1;
  field *yy = y->y + incy + 1;
  int off;
  int i, j;

  off = nx * ny;

  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      yx[i+j*incx] = x[i+j*nx];
      yy[i+j*incy] = x[i+j*nx+off];
    }
}

void
densefrom_edge2d(const edge2d *x, field *y)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int incx = nx + 1;
  int incy = nx + 2;
  const field *xx = x->x + incx + 1;
  const field *xy = x->y + incy + 1;
  int off;
  int i, j;

  off = nx * ny;

  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++) {
      y[i+j*nx] = xx[i+j*incx];
      y[i+j*nx+off] = xy[i+j*incy];
    }
}

void
unitindices_edge2d(int idx, const edge2d *x, int *i, int *j)
{
  int nx = x->gr->nx;
  int ny = x->gr->ny;
  int ii, jj;

  ii = idx;
  for(jj=1; jj<ny && ii>=nx; jj+=2)
    ii -= nx;
  assert(jj < ny);

  *i = ii;
  *j = jj;
}

void
unit_edge2d(int idx, edge2d *x)
{
  int nx = x->gr->nx;
  int incx = nx + 1;
  field *xx = x->x + incx + 1;
  real hx = x->gr->hx;
  real hy = x->gr->hy;
  field mx = hy / hx / 6.0;
  int i, j;

  unitindices_edge2d(idx, x, &i, &j);

  zero_edge2d(x);

  xx[i+j*incx] = sqrt(0.25 / mx);
}

field
massprod_xunit_edge2d(int i, int j, const edge2d *y)
{
  int nx = y->gr->nx;
  int ny = y->gr->ny;
  int incx = nx + 1;
  const field *yx = y->x + incx + 1;
  real hx = y->gr->hx;
  real hy = y->gr->hy;
  real mx = hy / hx / 6.0;
  field sum;

  assert(0 <= i && i < nx);
  assert(0 < j && j < ny);

  sum = sqrt(0.25 * mx) * (4.0 * yx[i+j*incx]
			   + (j > 0 ? yx[i+(j-1)*incx] : yx[i+(ny-1)*incx])
			   + yx[i+(j+1)*incx]);

#pragma omp barrier
  
  return sum;
}

void
add_xunit_edge2d(field alpha, int i, int j, edge2d *y)
{
  int nx = y->gr->nx;
  int incx = nx + 1;
  field *yx = y->x + incx + 1;
  real hx = y->gr->hx;
  real hy = y->gr->hy;
  real mx = hy / hx / 6.0;

  assert(0 <= i && i < nx);
  assert(0 < j && j < y->gr->ny);

#pragma omp single
  yx[i+j*incx] += alpha * sqrt(0.25 / mx);
}

void
buildhouseholder_edge2d(int idx, edge2d *x, field *tau)
{
  real alpha;
  field beta;
  int i, j;
  
  unitindices_edge2d(idx, x, &i, &j);

  alpha = sqrt(REAL(massprod_edge2d(x, x)));

  if(alpha == 0.0) {
    *tau = 0.0;
    return;
  }

  beta = massprod_xunit_edge2d(i, j, x);
  if(REAL(beta) < 0.0)
    alpha = -alpha;

  add_xunit_edge2d(alpha, i, j, x);

  *tau = 1.0 / (alpha * alpha + beta * alpha);
}

void
applyhouseholder_edge2d(const edge2d *v, field tau, edge2d *x)
{
  field beta;

  beta = massprod_edge2d(v, x);

  add_edge2d(-tau*beta, v, x);
}

void
orthonormalize_edge2d(int k, edge2d **x, field *tau, edge2d *v)
{
  field product;
  int i, j;
  int ix, iy;

  for(i=0; i<k; i++) {
    /* Create Householder vector for i-th input */
    buildhouseholder_edge2d(i, x[i], tau+i);

    /* Transform remaining input vectors */
    unitindices_edge2d(i, v, &ix, &iy);
    for(j=i+1; j<k; j++) {
      applyhouseholder_edge2d(x[i], tau[i], x[j]);
      
      product = massprod_xunit_edge2d(ix, iy, x[j]);
      add_xunit_edge2d(-product, ix, iy, x[j]);
    }
  }

  for(i=k; i-->0; ) {
    /* Set up i-th unit vector */
    unit_edge2d(i, v);

    /* Apply Householder reflections */
    for(j=i+1; j-->0; )
      applyhouseholder_edge2d(x[j], CONJ(tau[j]), v);

    /* Store the result */
    copy_edge2d(v, x[i]);
  }
}

void
pinvit_edge2d(int l,
	      int smoother_steps, int prec_steps, int gradient_steps,
	      real lambda, edge2d *e,
	      const matrix *Ae, edge2d **b, edge2d **x, edge2d **d,
	      const matrix *An, node2d **bg, node2d **xg, node2d **dg)
{
  int j;
  
  /* Compute the defect */
  zero_edge2d(b[l]);
  addeval_edge2d(1.0, 1.0-lambda, e, b[l]);

  /* Apply the multigrid preconditioner */
  zero_edge2d(x[l]);
  for(j=0; j<prec_steps; j++)
    vcycle_edge2d(l, smoother_steps, Ae, b, x, d);
	
  /* Subtract update from eigenvector approximation */
  add_edge2d(-1.0, x[l], e);
	  
  /* Reduce null-space components */
  zero_edge2d(d[l]);
  addeval_edge2d(0.0, 1.0, e, d[l]);
  zero_node2d(bg[l]);
  adjgradient_node2d(1.0, d[l], bg[l]);
  zero_node2d(xg[l]);
  for(j=0; j<gradient_steps; j++)
    vcycle_node2d(l, smoother_steps, An, bg, xg, dg);
  gradient_node2d(-1.0, xg[l], e);
}

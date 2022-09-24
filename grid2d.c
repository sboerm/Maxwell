
#include "grid2d.h"

#include "basic.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <cairo.h>
#include <cairo-pdf.h>

grid2d *
new_grid2d(int nx, int ny, real hx, real hy)
{
  grid2d *gr;
  int i, j;

  gr = (grid2d *) malloc(sizeof(grid2d));
  gr->hx = hx;
  gr->hy = hy;
  gr->nx = nx;
  gr->ny = ny;
  gr->xfactor = 1.0;
  gr->yfactor = 1.0;
  gr->eps = (real *) malloc(sizeof(real) * nx * ny);

  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      gr->eps[i+j*nx] = 1.0;
  
  return gr;
}

void
del_grid2d(grid2d *gr)
{
  free(gr->eps);
  gr->eps = 0;
  free(gr);
}

grid2d *
coarsen_grid2d(const grid2d *fg)
{
  grid2d *gr;
  int nx = fg->nx / 2;
  int ny = fg->ny / 2;
  real hx = 2.0 * fg->hx;
  real hy = 2.0 * fg->hy;
  int i, j;

  assert(fg->nx == 2 * nx);
  assert(fg->ny == 2 * ny);

  gr = new_grid2d(nx, ny, hx, hy);

  gr->xfactor = fg->xfactor;
  gr->yfactor = fg->yfactor;
  
  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      gr->eps[i+j*nx] = 0.25 * (fg->eps[2*i+2*j*2*nx] +
				fg->eps[2*i+1+2*j*2*nx] +
				fg->eps[2*i+(2*j+1)*2*nx] +
				fg->eps[2*i+1+(2*j+1)*2*nx]);

  return gr;
}

void
draw2d_grid2d(const grid2d *gr, const char *filename)
{
  cairo_t *cr;
  cairo_surface_t *pdf;
  cairo_surface_t *png;
  int nx = gr->nx;
  int ny = gr->ny;
  real hx = gr->hx;
  real hy = gr->hy;
  const real *eps = gr->eps;
  const char *c;
  real maxeps;
  int i, j;

  cr = 0;
  png = 0;

#pragma omp single
  {
    for(c=filename; *c; c++)
      ;
    if(filename+4 <= c && strncmp(c-4, ".png", 4) == 0) {
      if(nx * hx >= ny * hy) {
	png = cairo_image_surface_create(CAIRO_FORMAT_RGB24, 512, 512.0 * (ny * hy) / (nx * hx));
	cr = cairo_create(png);
	cairo_scale(cr, 512.0 / (nx * hx), 512.0 / (nx * hx));
	cairo_set_line_width(cr, nx * hx * 0.001);
      }
      else {
	png = cairo_image_surface_create(CAIRO_FORMAT_RGB24, 512.0 * (nx * hx) / (ny * hy), 512);
	cr = cairo_create(png);
	cairo_scale(cr, 512.0 / (ny * hy), 512.0 / (ny * hy));
	cairo_set_line_width(cr, ny * hy * 0.001);
      }
    }
    else {
      if(nx * hx >= ny * hy) {
	pdf = cairo_pdf_surface_create(filename, 400.0, 400.0 * (ny * hy) / (nx * hx));
	cr = cairo_create(pdf);
	cairo_surface_destroy(pdf);
	cairo_scale(cr, 400.0 / (nx * hx), 400.0 / (nx * hx));
	cairo_set_line_width(cr, nx * hx * 0.001);
      }
      else {
	pdf = cairo_pdf_surface_create(filename, 400.0 * (nx * hx) / (ny * hy), 400.0);
	cr = cairo_create(pdf);
	cairo_surface_destroy(pdf);
	cairo_scale(cr, 400.0 / (ny * hy), 400.0 / (ny * hy));
	cairo_set_line_width(cr, ny * hy * 0.001);
      }
    }

    maxeps = eps[0];
    for(j=0; j<ny; j++)
      for(i=0; i<nx; i++)
	if(eps[i+j*nx] > maxeps)
	  maxeps = eps[i+j*nx];
    
    for(j=0; j<ny; j++)
      for(i=0; i<nx; i++) {
	cairo_rectangle(cr, i*hx, j*hy, hx, hy);
	cairo_set_source_rgb(cr, eps[i+j*nx] / maxeps, 0.0, 1.0 - eps[i+j*nx] / maxeps);
	cairo_fill(cr);
      }
    
    if(png) {
      cairo_surface_write_to_png(png, filename);
      cairo_surface_destroy(png);
    }
    cairo_destroy(cr);
  }
}

/* ============================================================
 * Permittivity patterns
 * ============================================================ */

epspattern *
new_epspattern(real eps_base, int rectangles, int circles)
{
  epspattern *pat;

  pat = (epspattern *) malloc(sizeof(epspattern));
  pat->rectangle = (real *) malloc(sizeof(double) * 5 * rectangles);
  pat->rectangles = rectangles;
  pat->circle = (real *) malloc(sizeof(double) * 4 * circles);
  pat->circles = circles;
  pat->eps_base = eps_base;

  return pat;
}

void
del_epspattern(epspattern *pat)
{
  free(pat->circle);
  free(pat->rectangle);
  free(pat);
}

void
setpattern_grid2d(const epspattern *pat, grid2d *gr)
{
  const real *rectangle = pat->rectangle;
  const real *circle = pat->circle;
  int rectangles = pat->rectangles;
  int circles = pat->circles;
  real *eps = gr->eps;
  int nx = gr->nx;
  int ny = gr->ny;
  real hx = gr->hx;
  real hy = gr->hy;
  real x, y;
  int i, j, k;

  if(pat == 0)
    return;

  for(j=0; j<ny; j++)
    for(i=0; i<nx; i++)
      eps[i+j*nx] = 1.0 / pat->eps_base;
  
  for(k=0; k<rectangles; k++) {
    for(j=0; j<ny; j++) {
      y = j * hy;
      for(i=0; i<nx; i++) {
	x = i * hx;

	if(rectangle[5*k+1] <= x && x <= rectangle[5*k+1] + rectangle[5*k+3] &&
	   rectangle[5*k+2] <= y && y <= rectangle[5*k+2] + rectangle[5*k+4])
	  eps[i+j*nx] = 1.0 / rectangle[5*k];
      }
    }
  }

  for(k=0; k<circles; k++) {
    for(j=0; j<ny; j++) {
      y = j * hy;
      for(i=0; i<nx; i++) {
	x = i * hx;

	if((x - circle[4*k+1]) * (x - circle[4*k+1]) +
	   (y - circle[4*k+2]) * (y - circle[4*k+2]) <= circle[4*k+3] * circle[4*k+3])
	  eps[i+j*nx] = 1.0 / circle[4*k];
      }
    }
  }
}

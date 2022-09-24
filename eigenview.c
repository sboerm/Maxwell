
/* ============================================================
 * eigenview.c
 * Written 2022 by Steffen Boerm <boerm@math.uni-kiel.de>
 * All rights reserved.
 * ============================================================ */

#include <gtk/gtk.h>
#include <cairo-pdf.h>

#include <netcdf.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

/* ------------------------------------------------------------
 * Data for callback functions
 * ------------------------------------------------------------ */

typedef struct {
  int blochsteps;
  int eigenvalues;
  int gx;
  int gy;
  int rectangles;
  int circles;

  double *rectangle;
  double *circle;
  double mineps, maxeps;
  
  double *lambda;
  double maxlambda;

  double *xbloch, *ybloch;
  double *xfactorr, *xfactori, *yfactorr, *yfactori;
  
  double *exr, *exi, *eyr, *eyi;

  int blochmarked;
  int eigenmarked;
  double cr, ci;

  int displaymode;
  int epsmode;

  int nc_file;
  int nc_vectorxr, nc_vectorxi, nc_vectoryr, nc_vectoryi;
  
  GtkWidget *vector;
  GtkWidget *value;
  GtkSpinButton *blochspin;
  GtkSpinButton *eigenspin;
} eigendata;

/* ------------------------------------------------------------
 * Read eigenvector from NetCDF file
 * ------------------------------------------------------------ */

static void
read_eigenvector(eigendata *ed)
{
  int nc_file = ed->nc_file;
  int nc_vectorxr = ed->nc_vectorxr;
  int nc_vectorxi = ed->nc_vectorxi;
  int nc_vectoryr = ed->nc_vectoryr;
  int nc_vectoryi = ed->nc_vectoryi;
  size_t start[4], count[4];
  int info;

  if(ed->eigenmarked >= 0) {
    printf("Reading eigenvector %d in step %d\n",
	   ed->eigenmarked, ed->blochmarked);
    
    start[0] = ed->blochmarked;
    start[1] = ed->eigenmarked;
    start[2] = 0;
    start[3] = 0;
    count[0] = 1;
    count[1] = 1;
    count[2] = ed->gx;
    count[3] = ed->gy;
    info = nc_get_vara_double(nc_file, nc_vectorxr, start, count, ed->exr);
    assert(info == NC_NOERR);
    info = nc_get_vara_double(nc_file, nc_vectorxi, start, count, ed->exi);
    assert(info == NC_NOERR);
    info = nc_get_vara_double(nc_file, nc_vectoryr, start, count, ed->eyr);
    assert(info == NC_NOERR);
    info = nc_get_vara_double(nc_file, nc_vectoryi, start, count, ed->eyi);
    assert(info == NC_NOERR);
  }
}

/* ------------------------------------------------------------
 * Eigenvector visualization
 * ------------------------------------------------------------ */

static void
eigenvector_draw(cairo_t *cr, double width, double height,
		 const eigendata *ed)
{
  cairo_matrix_t G;
  int gx = ed->gx;
  int gy = ed->gy;
  double scale;
  double vx, vy, len1, len2, maxlen, xscale, y1scale, y2scale, tn;
  double vx0, vx1, vy0, vy1, vr, vi;
  int i, j;

  /* White background */
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
  cairo_fill(cr);

  /* Set up coordinate system */
  if(width / ed->gx > height / ed->gy) {
    cairo_translate(cr, 0.5 * (width - ed->gx * height / ed->gy), 0.0);
    cairo_scale(cr, height / ed->gy, height / ed->gy);
    scale = ed->gy;
  }
  else {
    cairo_translate(cr, 0.0, 0.5 * (height - ed->gy * width / ed->gx));
    cairo_scale(cr, width / ed->gx, width / ed->gx);
    scale = ed->gx;
  }

  switch(ed->epsmode) {
  default:
    /* Falls through */
  case 0: /* Permittivity visualization off */
    break;
    
  case 1: /* Permittivity visualization "Outlines" */
    if(ed->displaymode > 1) {
      cairo_set_line_width(cr, 0.2);
      
      vx = (1.0 - ed->mineps) / (ed->maxeps - ed->mineps);
      cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
      
      cairo_rectangle(cr, 0.0, 0.0, scale, scale);
      cairo_stroke(cr);
      
      for(i=0; i<ed->rectangles; i++) {
	vx = (ed->rectangle[5*i] - ed->mineps) / (ed->maxeps - ed->mineps);
	cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
	
	cairo_rectangle(cr,
			scale * ed->rectangle[5*i+1], scale * ed->rectangle[5*i+2],
			scale * ed->rectangle[5*i+3], scale * ed->rectangle[5*i+4]);
	cairo_stroke(cr);
      }
      
      for(i=0; i<ed->circles; i++) {
	vx = (ed->circle[4*i] - ed->mineps) / (ed->maxeps - ed->mineps);
	cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
	
	cairo_arc(cr, scale * ed->circle[4*i+1], scale * ed->circle[4*i+2],
		  scale * ed->circle[4*i+3], 0.0, 6.282);
	cairo_stroke(cr);
      }
    }
    break;
    
  case 2: /* Permittivity visualization "Filled" */
    if(ed->displaymode > 1) {
      vx = (1.0 - ed->mineps) / (ed->maxeps - ed->mineps);
      cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
      
      cairo_rectangle(cr, 0.0, 0.0, scale, scale);
      cairo_fill(cr);
      
      for(i=0; i<ed->rectangles; i++) {
	vx = (ed->rectangle[5*i] - ed->mineps) / (ed->maxeps - ed->mineps);
	cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
	
	cairo_rectangle(cr,
			scale * ed->rectangle[5*i+1], scale * ed->rectangle[5*i+2],
			scale * ed->rectangle[5*i+3], scale * ed->rectangle[5*i+4]);
	cairo_fill(cr);
      }
      
      for(i=0; i<ed->circles; i++) {
	vx = (ed->circle[4*i] - ed->mineps) / (ed->maxeps - ed->mineps);
	cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
	
	cairo_arc(cr, scale * ed->circle[4*i+1], scale * ed->circle[4*i+2],
		  scale * ed->circle[4*i+3], 0.0, 6.282);
	cairo_fill(cr);
      }
    }
    break;
  }
  
  /* Set the line width */
  cairo_set_line_width(cr, 0.05);

  switch(ed->displaymode) {
  case 0: /* "Absolute" visualization mode */
    maxlen = 0.0;
    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	len1 = vx * vx + vy * vy;

	vx = ed->exi[i+j*gx];
	vy = ed->eyi[i+j*gx];
	len1 += vx * vx + vy * vy;

	len1 = sqrt(len1);
	
	if(len1 > maxlen)
	  maxlen = len1;
      }

    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	len1 = vx * vx + vy * vy;

	vx = ed->exi[i+j*gx];
	vy = ed->eyi[i+j*gx];
	len1 += vx * vx + vy * vy;

	len1 = sqrt(len1) / maxlen;

	cairo_set_source_rgb(cr, 1.0, 1.0-len1, 1.0-len1);

	cairo_rectangle(cr, i, j, 1.0, 1.0);
	cairo_fill(cr);
      }
    break;
      
  case 1: /* "Electrical" visualization mode */
    maxlen = 0.0;
    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	/* Real part of the curl */
	vx0 = ed->exr[i+j*gx];
	vy0 = ed->eyr[i+j*gx];
	vx1 = (j+1 < gy ?
	       ed->exr[i+(j+1)*gx] :
	       ed->exr[i+0*gx] * ed->xfactorr[ed->blochmarked]
	       - ed->exi[i+0*gx] * ed->xfactori[ed->blochmarked]);
	vy1 = (i+1 < gx ?
	       ed->eyr[(i+1)+j*gx] :
	       ed->eyr[0+j*gx] * ed->yfactorr[ed->blochmarked]
	       - ed->eyi[0+j*gx] * ed->yfactori[ed->blochmarked]);
	vr = vx0 + vy1 - vx1 - vy0;

	/* Imaginary part of the curl */
	vx0 = ed->exi[i+j*gx];
	vy0 = ed->eyi[i+j*gx];
	vx1 = (j+1 < gy ?
	       ed->exi[i+(j+1)*gx] :
	       ed->exi[i+0*gx] * ed->xfactorr[ed->blochmarked]
	       + ed->exr[i+0*gx] * ed->xfactori[ed->blochmarked]);
	vy1 = (i+1 < gx ?
	       ed->eyi[(i+1)+j*gx] :
	       ed->eyi[0+j*gx] * ed->yfactorr[ed->blochmarked]
	       + ed->eyr[0+j*gx] * ed->yfactori[ed->blochmarked]);
	vi = vx0 + vy1 - vx1 - vy0;

	/* Compute absolute value */
	len1 = sqrt(vr * vr + vi * vi);

	if(len1 > maxlen)
	  maxlen = len1;
      }

    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	/* Real part of the curl */
	vx0 = ed->exr[i+j*gx];
	vy0 = ed->eyr[i+j*gx];
	vx1 = (j+1 < gy ?
	       ed->exr[i+(j+1)*gx] :
	       ed->exr[i+0*gx] * ed->xfactorr[ed->blochmarked]
	       - ed->exi[i+0*gx] * ed->xfactori[ed->blochmarked]);
	vy1 = (i+1 < gx ?
	       ed->eyr[(i+1)+j*gx] :
	       ed->eyr[0+j*gx] * ed->yfactorr[ed->blochmarked]
	       - ed->eyi[0+j*gx] * ed->yfactori[ed->blochmarked]);
	vr = vx0 + vy1 - vx1 - vy0;

	/* Imaginary part of the curl */
	vx0 = ed->exi[i+j*gx];
	vy0 = ed->eyi[i+j*gx];
	vx1 = (j+1 < gy ?
	       ed->exi[i+(j+1)*gx] :
	       ed->exi[i+0*gx] * ed->xfactorr[ed->blochmarked]
	       + ed->exr[i+0*gx] * ed->xfactori[ed->blochmarked]);
	vy1 = (i+1 < gx ?
	       ed->eyi[(i+1)+j*gx] :
	       ed->eyi[0+j*gx] * ed->yfactorr[ed->blochmarked]
	       + ed->eyr[0+j*gx] * ed->yfactori[ed->blochmarked]);
	vi = vx0 + vy1 - vx1 - vy0;

	/* Compute absolute value */
	len1 = sqrt(vr * vr + vi * vi);

	/* Scale it */
	len1 /= maxlen;

	cairo_set_source_rgb(cr, 1.0, 1.0-len1, 1.0-len1);

	cairo_rectangle(cr, i, j, 1.0, 1.0);
	cairo_fill(cr);
      }
    break;
      
  default:
    /* Falls through */
  case 2: /* "Real" visualization mode */
    maxlen = 0.0;
    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	len1 = sqrt(vx * vx + vy * vy);
	
	vx = ed->exi[i+j*gx];
	vy = ed->eyi[i+j*gx];
	len2 = sqrt(vx * vx + vy * vy);
	
	if(len1 > maxlen)
	  maxlen = len1;
	if(len2 > maxlen)
	  maxlen = len2;
      }
    
    for(j=0; j<gy; j++) {
      for(i=0; i<gx; i++) {
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	
	cairo_save(cr);
	
	/* Givens rotation to save work */
	if(fabs(vx) > fabs(vy)) {
	  tn = vy / fabs(vx);
	  G.xx = (vx >= 0.0 ?
		  1.0 / sqrt(1.0 + tn * tn) :
		  -1.0 / sqrt(1.0 + tn * tn));
	  G.yx = tn / sqrt(1.0 + tn * tn);
	}
	else if(fabs(vy) > 0.0) {
	  tn = vx / fabs(vy);
	  G.yx = (vy >= 0.0 ?
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
	G.x0 = i + 0.5;
	G.y0 = j + 0.5;
	cairo_transform(cr, &G);
	
	/* Draw an arrow */
	xscale = 0.5;
	y1scale = sqrt(vx * vx + vy * vy) / maxlen;
	y2scale = 0.5;
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
	
	/* Restore un-rotated coordinate system */
	cairo_restore(cr);
      }
    }
    break;
    
  case 3: /* "Real/Imag" visualization mode */
    maxlen = 0.0;
    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	len1 = sqrt(vx * vx + vy * vy);
	
	vx = ed->exi[i+j*gx];
	vy = ed->eyi[i+j*gx];
	len2 = sqrt(vx * vx + vy * vy);
	
	if(len1 > maxlen)
	  maxlen = len1;
	if(len2 > maxlen)
	  maxlen = len2;
      }
    
    for(j=0; j<gy; j++) {
      for(i=0; i<gx; i++) {
	vx = ed->exi[i+j*gx];
	vy = ed->eyi[i+j*gx];
	
	cairo_save(cr);
	
	/* Givens rotation to save work */
	if(fabs(vx) > fabs(vy)) {
	  tn = vy / fabs(vx);
	  G.xx = (vx >= 0.0 ?
		  1.0 / sqrt(1.0 + tn * tn) :
		  -1.0 / sqrt(1.0 + tn * tn));
	  G.yx = tn / sqrt(1.0 + tn * tn);
	}
	else if(fabs(vy) > 0.0) {
	  tn = vx / fabs(vy);
	  G.yx = (vy >= 0.0 ?
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
	G.x0 = i + 0.5;
	G.y0 = j + 0.5;
	cairo_transform(cr, &G);
	
	/* Draw an arrow */
	xscale = 0.5;
	y1scale = sqrt(vx * vx + vy * vy) / maxlen;
	y2scale = 0.5;
	cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
	cairo_move_to(cr, -0.2 * xscale, -y1scale);
	cairo_line_to(cr,  0.2 * xscale, -y1scale);
	cairo_line_to(cr,  0.2 * xscale, 0.5 * y1scale);
	cairo_line_to(cr,  0.5 * xscale, 0.5 * y1scale);
	cairo_line_to(cr,           0.0, 0.5 * (y1scale + y2scale));
	cairo_line_to(cr, -0.5 * xscale, 0.5 * y1scale);
	cairo_line_to(cr, -0.2 * xscale, 0.5 * y1scale);
	cairo_close_path(cr);
	cairo_fill(cr);
	
	/* Restore un-rotated coordinate system */
	cairo_restore(cr);
	
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	
	cairo_save(cr);
	
	/* Givens rotation to save work */
	if(fabs(vx) > fabs(vy)) {
	  tn = vy / fabs(vx);
	  G.xx = (vx >= 0.0 ?
		  1.0 / sqrt(1.0 + tn * tn) :
		  -1.0 / sqrt(1.0 + tn * tn));
	  G.yx = tn / sqrt(1.0 + tn * tn);
	}
	else if(fabs(vy) > 0.0) {
	  tn = vx / fabs(vy);
	  G.yx = (vy >= 0.0 ?
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
	G.x0 = i + 0.5;
	G.y0 = j + 0.5;
	cairo_transform(cr, &G);
	
	/* Draw an arrow */
	xscale = 0.5;
	y1scale = sqrt(vx * vx + vy * vy) / maxlen;
	y2scale = 0.5;
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
	
	/* Restore un-rotated coordinate system */
	cairo_restore(cr);
      }
    }
    break;

  case 4: /* "Animated" visualization mode */
    maxlen = 0.0;
    for(j=0; j<gy; j++)
      for(i=0; i<gx; i++) {
	vx = ed->exr[i+j*gx];
	vy = ed->eyr[i+j*gx];
	len1 = vx * vx + vy * vy;
	
	vx = ed->exi[i+j*gx];
	vy = ed->eyi[i+j*gx];
	len1 += vx * vx + vy * vy;

	len1 = sqrt(len1);
	
	if(len1 > maxlen)
	  maxlen = len1;
      }
    
    for(j=0; j<gy; j++) {
      for(i=0; i<gx; i++) {
	vx = ed->cr * ed->exr[i+j*gx] - ed->ci * ed->exi[i+j*gx];
	vy = ed->cr * ed->eyr[i+j*gx] - ed->ci * ed->eyi[i+j*gx];
	
	cairo_save(cr);
	
	/* Givens rotation to save work */
	if(fabs(vx) > fabs(vy)) {
	  tn = vy / fabs(vx);
	  G.xx = (vx >= 0.0 ?
		  1.0 / sqrt(1.0 + tn * tn) :
		  -1.0 / sqrt(1.0 + tn * tn));
	  G.yx = tn / sqrt(1.0 + tn * tn);
	}
	else if(fabs(vy) > 0.0) {
	  tn = vx / fabs(vy);
	  G.yx = (vy >= 0.0 ?
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
	G.x0 = i + 0.5;
	G.y0 = j + 0.5;
	cairo_transform(cr, &G);
	
	/* Draw an arrow */
	xscale = 0.5;
	y1scale = sqrt(vx * vx + vy * vy) / maxlen;
	y2scale = 0.5;
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
	
	/* Restore un-rotated coordinate system */
	cairo_restore(cr);
      }
    }
  }

  switch(ed->epsmode) {
  default:
    /* Falls through */
  case 0: /* Permittivity visualization off */
    break;
    
  case 1: /* Permittivity visualization "Outlines" */
    if(ed->displaymode < 2) {
      cairo_set_line_width(cr, 0.2);
      
      vx = (1.0 - ed->mineps) / (ed->maxeps - ed->mineps);
      cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
      
      cairo_rectangle(cr, 0.0, 0.0, scale, scale);
      cairo_stroke(cr);
      
      for(i=0; i<ed->rectangles; i++) {
	vx = (ed->rectangle[5*i] - ed->mineps) / (ed->maxeps - ed->mineps);
	cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
	
	cairo_rectangle(cr,
			scale * ed->rectangle[5*i+1], scale * ed->rectangle[5*i+2],
			scale * ed->rectangle[5*i+3], scale * ed->rectangle[5*i+4]);
	cairo_stroke(cr);
      }
      
      for(i=0; i<ed->circles; i++) {
	vx = (ed->circle[4*i] - ed->mineps) / (ed->maxeps - ed->mineps);
	cairo_set_source_rgb(cr, vx, 0.0, 1.0-vx);
	
	cairo_arc(cr, scale * ed->circle[4*i+1], scale * ed->circle[4*i+2],
		  scale * ed->circle[4*i+3], 0.0, 6.282);
	cairo_stroke(cr);
      }
    }
    break;
    
  case 2: /* Permittivity visualization "Filled" */
    break;
  }
}

static gboolean
vector_draw(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  eigendata *ed = (eigendata *) data;
  double width, height;

  /* Obtain size of widget */
  width = gtk_widget_get_allocated_width(widget);
  height = gtk_widget_get_allocated_height(widget);

  /* Draw eigenvector */
  eigenvector_draw(cr, width, height, ed);

  /* Event processing stops here */
  return TRUE;
}

/* Switch to "Absolute" visualization */
static gboolean
vector_absolute(GtkButton *widget, gpointer *data)
{
  eigendata *ed = (eigendata *) data;

  ed->displaymode = 0;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch to "Electrical" visualization */
static gboolean
vector_electrical(GtkButton *widget, gpointer *data)
{
  eigendata *ed = (eigendata *) data;

  ed->displaymode = 1;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch to "Real" visualization */
static gboolean
vector_real(GtkButton *widget, gpointer data)
{
  eigendata *ed = (eigendata *) data;

  ed->displaymode = 2;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch to "Real/Imag" visualization */
static gboolean
vector_realimag(GtkButton *widget, gpointer data)
{
  eigendata *ed = (eigendata *) data;

  ed->displaymode = 3;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch to "Animated" visualization */
static gboolean
vector_animated(GtkButton *widget, gpointer data)
{
  eigendata *ed = (eigendata *) data;

  ed->displaymode = 4;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch permittivity off */
static gboolean
vector_eps_off(GtkButton *widget, gpointer *data)
{
  eigendata *ed = (eigendata *) data;

  ed->epsmode = 0;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch permittivity to "Outlines" */
static gboolean
vector_eps_outlines(GtkButton *widget, gpointer *data)
{
  eigendata *ed = (eigendata *) data;

  ed->epsmode = 1;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Switch permittivity to "Filled" */
static gboolean
vector_eps_filled(GtkButton *widget, gpointer *data)
{
  eigendata *ed = (eigendata *) data;

  ed->epsmode = 2;

  gtk_widget_queue_draw(ed->vector);

  return TRUE;
}

/* Bloch step has been changed */
static void
vector_bloch_changed(GtkSpinButton *widget, gpointer data)
{
  eigendata *ed = (eigendata *) data;

  ed->blochmarked = gtk_spin_button_get_value(widget);

  read_eigenvector(ed);

  gtk_widget_queue_draw(ed->vector);
  gtk_widget_queue_draw(ed->value);
}

/* Eigenvalue has been changed */
static void
vector_eigenvalue_changed(GtkSpinButton *widget, gpointer data)
{
  eigendata *ed = (eigendata *) data;

  ed->eigenmarked = gtk_spin_button_get_value(widget);

  read_eigenvector(ed);

  gtk_widget_queue_draw(ed->vector);
  gtk_widget_queue_draw(ed->value);
}

/* Save button clicked */
static gboolean
vector_save_clicked(GtkButton *widget, gpointer *data)
{
  eigendata *ed = (eigendata *) data;
  GtkWidget *dialog;
  char suggestion[40];
  cairo_surface_t *surface;
  cairo_t *cr;
  char *filename;
  gint res;

  dialog = gtk_file_chooser_dialog_new("Save Image",
				       GTK_WINDOW(ed->vector),
				       GTK_FILE_CHOOSER_ACTION_SAVE,
				       "_Cancel",
				       GTK_RESPONSE_CANCEL,
				       "_Save",
				       GTK_RESPONSE_ACCEPT,
				       NULL);

  snprintf(suggestion, 40, "eigenvector_%d_%d.pdf", ed->blochmarked, ed->eigenmarked);
  gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog),
				    suggestion);

  res = gtk_dialog_run(GTK_DIALOG(dialog));

  if(res == GTK_RESPONSE_ACCEPT) {
    filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

    surface = cairo_pdf_surface_create(filename, 800.0, 800.0);
    cr = cairo_create(surface);

    eigenvector_draw(cr, 800.0, 800.0, ed);

    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    g_free(filename);
  }

  gtk_widget_destroy(dialog);

  

  return TRUE;
}

/* Update phase variable for "Animated" visualization mode */
static gboolean
vector_animate(GtkWidget *widget, GdkFrameClock *clock, gpointer data)
{
  eigendata *ed = (eigendata *) data;
  double t = gdk_frame_clock_get_frame_time(clock) * 0.000001;

  /* Compute the current phase */
  ed->cr = cos(M_PI * t);
  ed->ci = sin(M_PI * t);

  /* Redraw eigenvector representation if animated */
  if(ed->displaymode == 4)
    gtk_widget_queue_draw(widget);

  return G_SOURCE_CONTINUE;
}

/* Draw Bloch phase parameters */
static gboolean
vector_phase(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  eigendata *ed = (eigendata *) data;
  double width, height;

  /* Obtain size of widget */
  width = gtk_widget_get_allocated_width(widget);
  height = gtk_widget_get_allocated_height(widget);

  /* White background */
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
  cairo_fill(cr);

  /* Draw circles */
  cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
  cairo_arc(cr, 0.25 * width, 0.5 * height, 0.2 * width, 0.0, 6.282);
  cairo_stroke(cr);
  cairo_arc(cr, 0.75 * width, 0.5 * height, 0.2 * width, 0.0, 6.282);
  cairo_stroke(cr);

  /* Draw parameters */
  cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
  cairo_arc(cr,
	    0.25 * width + 0.2 * width * ed->xfactorr[ed->blochmarked],
	    0.5 * height - 0.2 * width * ed->xfactori[ed->blochmarked],
	    0.04 * width, 0.0, 6.282);
  cairo_fill(cr);
  cairo_arc(cr,
	    0.75 * width + 0.2 * width * ed->yfactorr[ed->blochmarked],
	    0.5 * height - 0.2 * width * ed->yfactori[ed->blochmarked],
	    0.04 * width, 0.0, 6.282);
  cairo_fill(cr);

  return TRUE;
}

/* Set up "eigenvector" window */
static void
build_vector_window(eigendata *ed)
{
  GtkWidget *window, *hbox, *vbox;
  GtkWidget *widget, *frame, *radiobox;

  /* Create eigenvector window */
  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  ed->vector = window;

  /* Set title */
  gtk_window_set_title(GTK_WINDOW(window), "Bloch eigenvector");

  /* Set initial size */
  gtk_window_set_default_size(GTK_WINDOW(window), 1000, 800);

  /* Destroying the window exits the main loop, ending the program */
  g_signal_connect(ed->vector, "destroy", G_CALLBACK(gtk_main_quit), NULL);

  /* Horizontal box for the eigenvector and the control panel */
  hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);

  /* Put the box into the window */
  gtk_container_add(GTK_CONTAINER(window), hbox);

  /* Create a drawing area for the eigenvector */
  widget = gtk_drawing_area_new();

  /* Set draw callback */
  g_signal_connect(widget, "draw", G_CALLBACK(vector_draw), ed);

  /* Add animation callback */
  gtk_widget_add_tick_callback(widget, vector_animate, ed, 0);

  /* Put the drawing area into the horizontal box */
  gtk_box_pack_start(GTK_BOX(hbox), widget, TRUE, TRUE, 4);

  /* Create settings box */
  vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);

  /* Create "Visualization" frame */
  frame = gtk_frame_new("Visualization");

  /* Box of radio buttons */
  radiobox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);

  /* Create "Absolute" radio button */
  widget = gtk_radio_button_new_with_label(0, "Absolute");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_absolute), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);
  
  /* Create "Electrical" radio button */
  widget = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(widget), "Electrical");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_electrical), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);

  /* Create "Real" radio button */
  widget = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(widget), "Real");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_real), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);

  /* Create "Real/Imag" radio button */
  widget = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(widget), "Real/Imag");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_realimag), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);

  /* Create "Animated" radio button */
  widget = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(widget), "Animated");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_animated), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);

  /* Put radio box into frame */
  gtk_container_add(GTK_CONTAINER(frame), radiobox);

  /* Put frame into the settings box */
  gtk_box_pack_start(GTK_BOX(vbox), frame, FALSE, FALSE, 4);

  /* Create "Permittivity" frame */
  frame = gtk_frame_new("Permittivity");

  /* Box of radio buttons */
  radiobox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);

  /* Create "Off" radio button */
  widget = gtk_radio_button_new_with_label(0, "Off");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_eps_off), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);
  
  /* Create "Outlines" radio button */
  widget = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(widget), "Outlines");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_eps_outlines), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);

  /* Create "Filled" radio button */
  widget = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(widget), "Filled");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_eps_filled), ed);
  gtk_box_pack_start(GTK_BOX(radiobox), widget, FALSE, FALSE, 2);

  /* Put radio box into frame */
  gtk_container_add(GTK_CONTAINER(frame), radiobox);

  /* Put frame into the settings box */
  gtk_box_pack_start(GTK_BOX(vbox), frame, FALSE, FALSE, 4);

  /* Create "Bloch step" frame */
  frame = gtk_frame_new("Bloch step");

  /* Create spin button for the Bloch step */
  widget = gtk_spin_button_new_with_range(0, ed->blochsteps-1, 1);
  ed->blochspin = GTK_SPIN_BUTTON(widget);
  g_signal_connect(widget, "value-changed", G_CALLBACK(vector_bloch_changed), ed);
  gtk_container_add(GTK_CONTAINER(frame), widget);

  /* Put frame into the settings box */
  gtk_box_pack_start(GTK_BOX(vbox), frame, FALSE, FALSE, 4);

  /* Create "Eigenvalue" frame */
  frame = gtk_frame_new("Eigenvalue");

  /* Create spin button for the eigenvalue */
  widget = gtk_spin_button_new_with_range(0, ed->eigenvalues-1, 1);
  ed->eigenspin = GTK_SPIN_BUTTON(widget);
  g_signal_connect(widget, "value-changed", G_CALLBACK(vector_eigenvalue_changed), ed);
  gtk_container_add(GTK_CONTAINER(frame), widget);

  /* Put frame into the settings box */
  gtk_box_pack_start(GTK_BOX(vbox), frame, FALSE, FALSE, 4);

  /* Create "Save" button */
  widget = gtk_button_new_with_label("Save");
  g_signal_connect(widget, "clicked", G_CALLBACK(vector_save_clicked), ed);

  /* Put button into the settings box */
  gtk_box_pack_start(GTK_BOX(vbox), widget, FALSE, FALSE, 4);

  /* Create "Bloch phase" drawing area */
  widget = gtk_drawing_area_new();
  g_signal_connect(widget, "draw", G_CALLBACK(vector_phase), ed);
  gtk_widget_set_size_request(widget, 100, 50);

  /* Put drawing area into the settings box */
  gtk_box_pack_start(GTK_BOX(vbox), widget, FALSE, TRUE, 4);
  
  /* Put settings box into horizontal box */
  gtk_box_pack_start(GTK_BOX(hbox), vbox, FALSE, FALSE, 4);
}

/* ------------------------------------------------------------
 * Eigenvalue visualization
 * ------------------------------------------------------------ */

static gboolean
value_draw(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  eigendata *ed = (eigendata *) data;
  int blochsteps = ed->blochsteps;
  int eigenvalues = ed->eigenvalues;
  double maxlambda = ed->maxlambda;
  double width, height;
  int i, j;

  /* Obtain size of widget */
  width = gtk_widget_get_allocated_width(widget);
  height = gtk_widget_get_allocated_height(widget);

  /* Draw eigenvalue curves */
  for(j=0; j<eigenvalues; j++) {
    switch(j % 5) {
    case 0:
      cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
      break;
    case 1:
      cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
      break;
    case 2:
      cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);
      break;
    case 3:
      cairo_set_source_rgb(cr, 0.0, 0.0, 1.0);
      break;
    case 4:
      cairo_set_source_rgb(cr, 1.0, 0.0, 1.0);
      break;
    }

    cairo_move_to(cr, 0.0, height-height*ed->lambda[j]/maxlambda);
    for(i=1; i<blochsteps; i++)
      cairo_line_to(cr, i*width/blochsteps, height-height*ed->lambda[j+eigenvalues*i]/maxlambda);
    cairo_stroke(cr);
  }

  /* If an eigenvalue is marked, draw a small circle around it */
  if(ed->eigenmarked >= 0) {
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_arc(cr, ed->blochmarked*width/blochsteps, height-height*ed->lambda[ed->eigenmarked+eigenvalues*ed->blochmarked]/maxlambda, 4.0, 0.0, 6.282);
    cairo_stroke(cr);
  }

  /* Event processing stops here */
  return TRUE;
}

/* Mark or unmark an eigenvalue per mouse click */
static gboolean
value_button_press(GtkWidget *widget, GdkEvent *event,
		   gpointer data)
{
  eigendata *ed = (eigendata *) data;
  GdkEventButton *event_button = &event->button;
  int blochsteps = ed->blochsteps;
  int eigenvalues = ed->eigenvalues;
  double maxlambda = ed->maxlambda;
  double width, height;
  int blochstep, eigenvalue;
  int i;

  /* Obtain size of widget */
  width = gtk_widget_get_allocated_width(widget);
  height = gtk_widget_get_allocated_height(widget);

  /* Compute Bloch step */
  blochstep = event_button->x * blochsteps / width;

  /* Look for matching eigenvalue */
  eigenvalue = -1;
  for(i=0; i<eigenvalues; i++)
    if(fabs(height - ed->lambda[i+blochstep*eigenvalues] * height / maxlambda
	    - event_button->y) < 3)
      eigenvalue = i;

  /* Store results */
  ed->blochmarked = blochstep;
  ed->eigenmarked = eigenvalue;

  /* Enqueue a redraw event */
  gtk_widget_queue_draw(widget);

  if(eigenvalue == -1) {
    /* Hide eigenvector window */
    gtk_widget_hide(ed->vector);
  }
  else {
    /* Read eigenvector from NetCDF file */
    read_eigenvector(ed);

    /* Show eigenvector window */
    gtk_widget_show_all(ed->vector);

    /* Update spin buttons */
    gtk_spin_button_set_value(ed->blochspin, blochstep);
    gtk_spin_button_set_value(ed->eigenspin, eigenvalue);
  }
  
  return TRUE;
}

/* Set up "eigenvalue" window */
static void
build_value_window(eigendata *ed)
{
  GtkWidget *window, *widget;
  
  /* Create eigenvalue window */
  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  ed->value = window;

  /* Set window title */
  gtk_window_set_title(GTK_WINDOW(window), "Bloch eigenvalues");

  /* Set preferred initial size */
  gtk_window_set_default_size(GTK_WINDOW(window), 800, 600);

  /* Set preferred initial position */
  gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);

  /* Destroying the window exits the main loop, ending the program */
  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

  /* Create a drawing area */
  widget = gtk_drawing_area_new();
  
  /* Set the callback function for the "draw" event */
  g_signal_connect(widget, "draw", G_CALLBACK(value_draw), ed);

  /* Set the callback function for the "button-press" event */
  gtk_widget_add_events(widget, GDK_BUTTON_PRESS_MASK);
  g_signal_connect(widget, "button-press-event",
		   G_CALLBACK(value_button_press), ed);

  /* Put it into the window */
  gtk_container_add(GTK_CONTAINER(window), widget);
}

/* ------------------------------------------------------------
 * Main program
 * ------------------------------------------------------------ */

int
main(int argc, char **argv)
{
  eigendata *ed;
  int nc_file, nc_blochsteps, nc_eigenvalues, nc_lambda;
  int nc_graphx, nc_graphy;
  int nc_rectangles, nc_circles, nc_rectangle, nc_circle;
  int nc_xbloch, nc_ybloch;
  int nc_xfactorr, nc_xfactori, nc_yfactorr, nc_yfactori;
  int nc_vectorxr, nc_vectorxi, nc_vectoryr, nc_vectoryi;
  size_t blochsteps, eigenvalues, gx, gy, rectangles, circles;
  size_t start[2], count[2];
  double maxlambda, mineps, maxeps;
  int info;
  int i, j;

  /* Initialize GTK+ */
  gtk_init(&argc, &argv);

  /* We need a file name */
  if(argc < 2) {
    printf("%s [file name], please.\n", argv[0]);
    return 1;
  }
  
  /* Open NetCDF file */
  printf("Opening file \"%s\"\n", argv[1]);
  info = nc_open(argv[1], 0, &nc_file);
  if(info != NC_NOERR) {
    printf("Failed with error %d: \"%s\"\n",
	   info, nc_strerror(info));
    return 2;
  }

  /* Get number of Bloch steps */
  info = nc_inq_dimid(nc_file, "blochsteps", &nc_blochsteps);
  assert(info == NC_NOERR);

  info = nc_inq_dimlen(nc_file, nc_blochsteps, &blochsteps);
  assert(info == NC_NOERR);

  /* Report progress */
  printf("%zu Bloch steps\n", blochsteps);
  
  /* Get number of eigenvalues */
  info = nc_inq_dimid(nc_file, "eigenvalues", &nc_eigenvalues);
  assert(info == NC_NOERR);

  info = nc_inq_dimlen(nc_file, nc_eigenvalues, &eigenvalues);
  assert(info == NC_NOERR);

  /* Report progress */
  printf("%zu eigenvalues\n", eigenvalues);
  
  /* Get horizontal resolution of eigenvectors */
  info = nc_inq_dimid(nc_file, "graphx", &nc_graphx);
  assert(info == NC_NOERR);

  info = nc_inq_dimlen(nc_file, nc_graphx, &gx);
  assert(info == NC_NOERR);

  /* Get vertical resolution of eigenvectors */
  info = nc_inq_dimid(nc_file, "graphy", &nc_graphy);
  assert(info == NC_NOERR);

  info = nc_inq_dimlen(nc_file, nc_graphy, &gy);
  assert(info == NC_NOERR);

  /* Report progress */
  printf("Eigenvalue resolution %zu x %zu\n", gx, gy);

  /* Get numbers of rectangles and circles */
  info = nc_inq_dimid(nc_file, "rectangles", &nc_rectangles);
  if(info != NC_NOERR) {
    printf("Error %d when accessing permittivity data: \"%s\"\n",
	   info, nc_strerror(info));
    return 3;
  }

  info = nc_inq_dimlen(nc_file, nc_rectangles, &rectangles);
  assert(info == NC_NOERR);

  info = nc_inq_dimid(nc_file, "circles", &nc_circles);
  assert(info == NC_NOERR);

  info = nc_inq_dimlen(nc_file, nc_circles, &circles);
  assert(info == NC_NOERR);
  
  /* Report progress */
  printf("%zu rectangles, %zu circles\n",
	 rectangles / 5, circles / 4);

  /* Initialize "eigendata" structure */
  ed = (eigendata *) malloc(sizeof(eigendata));
  ed->blochsteps = blochsteps;
  ed->eigenvalues = eigenvalues;
  ed->gx = gx;
  ed->gy = gy;
  ed->rectangles = rectangles / 5;
  ed->circles = circles / 4;
  ed->rectangle = (double *) malloc(sizeof(double) * rectangles);
  ed->circle = (double *) malloc(sizeof(double) * circles);
  ed->lambda = (double *) malloc(sizeof(double) * blochsteps * eigenvalues);
  ed->xbloch = (double *) malloc(sizeof(double) * blochsteps);
  ed->ybloch = (double *) malloc(sizeof(double) * blochsteps);
  ed->xfactorr = (double *) malloc(sizeof(double) * blochsteps);
  ed->xfactori = (double *) malloc(sizeof(double) * blochsteps);
  ed->yfactorr = (double *) malloc(sizeof(double) * blochsteps);
  ed->yfactori = (double *) malloc(sizeof(double) * blochsteps);
  ed->exr = (double *) malloc(sizeof(double) * gx * gy);
  ed->exi = (double *) malloc(sizeof(double) * gx * gy);
  ed->eyr = (double *) malloc(sizeof(double) * gx * gy);
  ed->eyi = (double *) malloc(sizeof(double) * gx * gy);
  ed->blochmarked = 0;
  ed->eigenmarked = -1;
  ed->cr = 1.0;
  ed->ci = 0.0;
  ed->displaymode = 0;

  /* Read rectangles and circles */
  info = nc_inq_varid(nc_file, "rectangle", &nc_rectangle);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = rectangles;
  info = nc_get_vara_double(nc_file, nc_rectangle, start, count, ed->rectangle);
  assert(info == NC_NOERR);

  info = nc_inq_varid(nc_file, "circle", &nc_circle);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = circles;
  info = nc_get_vara_double(nc_file, nc_circle, start, count, ed->circle);
  assert(info == NC_NOERR);

  mineps = maxeps = 1.0;
  for(i=0; i<ed->rectangles; i++)
    if(ed->rectangle[5*i] < mineps)
      mineps = ed->rectangle[5*i];
    else if(ed->rectangle[5*i] > maxeps)
      maxeps = ed->rectangle[5*i];
  for(i=0; i<ed->circles; i++)
    if(ed->circle[4*i] < mineps)
      mineps = ed->circle[4*i];
    else if(ed->circle[4*i] > maxeps)
      maxeps = ed->circle[4*i];
  ed->mineps = mineps;
  ed->maxeps = maxeps;
  
  /* Report progress */
  printf("Permittivity data acquired, minimum %f, maximum %f\n",
	 mineps, maxeps);
  
  /* Read all eigenvalues */
  info = nc_inq_varid(nc_file, "lambda", &nc_lambda);
  assert(info == NC_NOERR);

  start[0] = 0;
  start[1] = 0;
  count[0] = blochsteps;
  count[1] = eigenvalues;
  info = nc_get_vara_double(nc_file, nc_lambda, start, count, ed->lambda);
  assert(info == NC_NOERR);

  /* Determine maximal eigenvalue */
  maxlambda = ed->lambda[0];
  for(i=0; i<eigenvalues; i++)
    for(j=0; j<blochsteps; j++)
      if(ed->lambda[i+j*eigenvalues] > maxlambda)
	maxlambda = ed->lambda[i+j*eigenvalues];
  ed->maxlambda = maxlambda;

  /* Report progress */
  printf("Eigenvalues acquired, maximum %f\n", maxlambda);

  /* Read Bloch parameters */
  info = nc_inq_varid(nc_file, "xbloch", &nc_xbloch);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = blochsteps;
  info = nc_get_vara_double(nc_file, nc_xbloch, start, count, ed->xbloch);

  info = nc_inq_varid(nc_file, "ybloch", &nc_ybloch);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = blochsteps;
  info = nc_get_vara_double(nc_file, nc_ybloch, start, count, ed->ybloch);

  info = nc_inq_varid(nc_file, "xfactorr", &nc_xfactorr);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = blochsteps;
  info = nc_get_vara_double(nc_file, nc_xfactorr, start, count, ed->xfactorr);

  info = nc_inq_varid(nc_file, "xfactori", &nc_xfactori);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = blochsteps;
  info = nc_get_vara_double(nc_file, nc_xfactori, start, count, ed->xfactori);
  
  info = nc_inq_varid(nc_file, "yfactorr", &nc_yfactorr);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = blochsteps;
  info = nc_get_vara_double(nc_file, nc_yfactorr, start, count, ed->yfactorr);

  info = nc_inq_varid(nc_file, "yfactori", &nc_yfactori);
  assert(info == NC_NOERR);

  start[0] = 0;
  count[0] = blochsteps;
  info = nc_get_vara_double(nc_file, nc_yfactori, start, count, ed->yfactori);

  (void) printf("Bloch parameters acquired\n");
  
  /* Obtain NetCDF variables for the eigenvectors */
  info = nc_inq_varid(nc_file, "vectorxr", &nc_vectorxr);
  assert(info == NC_NOERR);
  info = nc_inq_varid(nc_file, "vectorxi", &nc_vectorxi);
  assert(info == NC_NOERR);
  info = nc_inq_varid(nc_file, "vectoryr", &nc_vectoryr);
  assert(info == NC_NOERR);
  info = nc_inq_varid(nc_file, "vectoryi", &nc_vectoryi);
  assert(info == NC_NOERR);
  ed->nc_file = nc_file;
  ed->nc_vectorxr = nc_vectorxr;
  ed->nc_vectorxi = nc_vectorxi;
  ed->nc_vectoryr = nc_vectoryr;
  ed->nc_vectoryi = nc_vectoryi;

  (void) printf("Eigenvectors acquired\n");
  
  /* Create GTK+ windows */
  build_value_window(ed);
  build_vector_window(ed);

  (void) printf("GTK+ windows created\n");
  
  /* Make the eigenvalue window visible */
  gtk_widget_show_all(ed->value);

  /* Start GTK+ main loop */
  gtk_main();

  return 0;
}

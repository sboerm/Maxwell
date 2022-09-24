
#ifndef GRID2D_H
#define GRID2D_H

#include "settings.h"

typedef struct {
  /** @brief Meshwidth in x direction */
  real hx;

  /** @brief Meshwidth in y direction */
  real hy;
  
  /** @brief Number of intervals in x direction */
  int nx;

  /** @brief Number of intervals in y direction */
  int ny;

  /** @brief Bloch phase factor in x direction */
  real xbloch;

  /** @brief Bloch phase factor in y direction */
  real ybloch;
  
  /** @brief Periodicity factor in x direction */
  field xfactor;

  /** @brief Periodicity factor in y direction */
  field yfactor;
  
  /** @brief Dielectricity for every box */
  real *eps;
} grid2d;

grid2d *
new_grid2d(int nx, int ny, real hx, real hy);

void
del_grid2d(grid2d *gr);

grid2d *
coarsen_grid2d(const grid2d *fg);

void
draw2d_grid2d(const grid2d *gr, const char *filename);

/* ============================================================
 * Permittivity patterns
 * ============================================================ */

typedef struct {
  /** @brief Description of rectangles: reciprocal permittivity,
   *  left and lower boundary, width and height */
  real *rectangle;

  /** @brief Number of rectangles */
  int rectangles;

  /** @brief Description of circles: reciprocal permittivity,
   *  x and y of center, radius */
  real *circle;

  /** @brief Number of circles */
  int circles;

  /** @brief Permittivity of the surrounding material */
  real eps_base;
} epspattern;

epspattern *
new_epspattern(real eps_base, int rectangles, int circles);

void
del_epspattern(epspattern *pat);

void
setpattern_grid2d(const epspattern *pat, grid2d *gr);

#endif

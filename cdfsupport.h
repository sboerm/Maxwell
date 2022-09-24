
#ifndef CDFSUPPORT_H
#define CDFSUPPORT_H

typedef struct {
  int rows;
  int cols;
  
  int ncid;
  int ncdim[2];
  int ncvar;
} cdfmatrix;

cdfmatrix *
new_cdfmatrix(int rows, int cols, const char *matname, const char *filename);

void
del_cdfmatrix(cdfmatrix *cm);

#endif

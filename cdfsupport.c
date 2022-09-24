
#include "cdfsupport.h"

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>

cdfmatrix *
new_cdfmatrix(int rows, int cols, const char *matname, const char *filename)
{
  cdfmatrix *cm;
  int info;

  cm = (cdfmatrix *) malloc(sizeof(cdfmatrix));

  cm->rows = rows;
  cm->cols = cols;
  
  info = nc_create(filename, NC_CLOBBER | NC_64BIT_DATA, &cm->ncid);
  if(info != NC_NOERR)
    printf("NetCDF nc_create error code %d\n", info);

  info = nc_def_dim(cm->ncid, "cols", cols, cm->ncdim);
  if(info != NC_NOERR)
    printf("NetCDF nc_def_dim error code %d\n", info);
  
  info = nc_def_dim(cm->ncid, "rows", rows, cm->ncdim+1);
  if(info != NC_NOERR)
    printf("NetCDF nc_def_dim error code %d\n", info);

  info = nc_def_var(cm->ncid, matname, NC_DOUBLE, 2, cm->ncdim, &cm->ncvar);
  if(info != NC_NOERR)
    printf("NetCDF nc_def_var error code %d\n", info);

  nc_enddef(cm->ncid);

  return cm;
}

void
del_cdfmatrix(cdfmatrix *cm)
{
  nc_close(cm->ncid);
  cm->ncid = 0;

  free(cm);
}

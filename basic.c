
#include "basic.h"

#include <omp.h>
#include <assert.h>
#include <stdlib.h>

/* DEBUGGING */
#include <stdio.h>

real
reduce_sum_real(real x)
{
  int rank = omp_get_thread_num();
  int size = omp_get_num_threads();
  static real *shared = 0;
  int i;

  if(rank == 0) {
    assert(shared == 0);
    shared = (real *) malloc(sizeof(real) * size);
  }

#pragma omp barrier
  
  assert(shared != 0);
  
  shared[rank] = x;
  for(i=1; i<size; i*=2)
    ;

#pragma omp barrier
  
  for(; i>0; i/=2) {
    if(rank < i && rank+i < size)
      shared[rank] += shared[rank+i];
#pragma omp barrier
  }

  x = shared[0];

#pragma omp barrier
  
  if(rank == 0) {
    free(shared);
    shared = 0;
  }

#pragma omp barrier
  
  return x;
}

field
reduce_sum_field(field x)
{
  int rank = omp_get_thread_num();
  int size = omp_get_num_threads();
  static field *shared = 0;
  int i;

  if(rank == 0) {
    assert(shared == 0);
    shared = (field *) malloc(sizeof(field) * size);
  }

#pragma omp barrier

  assert(shared != 0);
  
  shared[rank] = x;
  for(i=1; i<size; i*=2)
    ;

#pragma omp barrier
  
  for(; i>0; i/=2) {
    if(rank < i && rank+i < size)
      shared[rank] += shared[rank+i];
#pragma omp barrier
  }

  x = shared[0];

#pragma omp barrier
  
  if(rank == 0) {
    free(shared);
    shared = 0;
  }

#pragma omp barrier
  
  return x;
}

real
reduce_max_real(real x)
{
  int rank = omp_get_thread_num();
  int size = omp_get_num_threads();
  static real *shared = 0;
  int i;

  if(rank == 0) {
    assert(shared == 0);
    shared = (real *) malloc(sizeof(real) * size);

    for(i=0; i<size; i++)
      shared[i] = 0.0;
  }
  
#pragma omp barrier
  
  assert(shared != 0);
  
  shared[rank] = x;
  for(i=1; i<size; i*=2)
    ;

#pragma omp barrier

  for(i=0; i<size; i++)
    if(shared[i] == 0.0)
      printf("%d uninitialized\n", i);

#pragma omp barrier
  
  for(; i>0; i/=2) {
    if(rank < i && rank+i < size)
      shared[rank] = (shared[rank] > shared[rank+i] ?
		      shared[rank] : shared[rank+i]);
#pragma omp barrier
  }

  x = shared[0];

#pragma omp barrier
  
  if(rank == 0) {
    free(shared);
    shared = 0;
  }

#pragma omp barrier
  
  return x;
}

field *
bcast_fieldptr(field *a)
{
  static field *shared = 0;

  if(a)
    shared = a;

#pragma omp barrier

  return shared;
}

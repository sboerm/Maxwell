
#ifndef BASIC_H
#define BASIC_H

#include "settings.h"

real
reduce_sum_real(real x);

field
reduce_sum_field(field x);

real
reduce_max_real(real x);

field *
bcast_fieldptr(field *a);

#endif

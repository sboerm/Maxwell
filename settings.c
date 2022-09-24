
#include "settings.h"

real
my_abs2(field x)
{
#ifdef FIELD_COMPLEX
  return REAL(x) * REAL(x) + IMAG(x) * IMAG(x);
#else
  return x * x;
#endif
}

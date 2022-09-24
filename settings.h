
#ifndef SETTINGS_H
#define SETTINGS_H

typedef double real;

#ifdef FIELD_COMPLEX
#include <complex.h>

typedef double complex field;

#define ABS(x) cabs(x)
#define ABS2(x) my_abs2(x)
#define CONJ(x) conj(x)
#define REAL(x) creal(x)
#define IMAG(x) cimag(x)
#define EXP(x) cexp(x)
#else
typedef double field;

#define ABS(x) fabs(x)
#define ABS2(x) my_abs2(x)
#define CONJ(x) (x)
#define REAL(x) (x)
#define IMAG(x) 0
#define EXP(x) exp(x)
#endif

real
my_abs2(field x);

#endif

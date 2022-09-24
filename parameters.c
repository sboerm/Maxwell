
#include "parameters.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

int
askforint(const char *question,
	  const char *envname,
	  int deflt)
{
  int res;
  char buf[80];
  char *env;

  env = getenv(envname);
  if(env && sscanf(env, "%d", &res) == 1)
    return res;

  (void) printf("%s (%d)\n", question, deflt);
  res = deflt;
  if(fgets(buf, 80, stdin))
    sscanf(buf, "%d", &res);

  return res;
}

char
askforchar(const char *question, const char *envname, const char *allowed, char deflt)
{
  char res;
  char buf[80];
  char *env;
  const char *c;

  env = getenv(envname);
  if(env && sscanf(env, "%c", &res) == 1)
    return tolower(res);

  do {
    (void) printf("%s (", question);
    res = tolower(deflt);
    for(c=allowed; *c; c++)
      if(*c == res)
	(void) printf("%c", toupper(*c));
      else
	(void) printf("%c", tolower(*c));
    (void) printf(")\n");
    if(fgets(buf, 80, stdin))
      sscanf(buf, " %c", &res);
    res = tolower(res);

    for(c=allowed; *c && *c!=res; c++)
      ;
  }
  while(!*c);

  return res;
}

real
askforreal(const char *question,
	   const char *envname,
	   real deflt)
{
  real res;
  char buf[80];
  char *env;

  env = getenv(envname);
  if(env && sscanf(env, "%lf", &res) == 1)
    return res;

  (void) printf("%s (%.3e)\n", question, deflt);
  res = deflt;
  if(fgets(buf, 80, stdin))
    sscanf(buf, "%lf", &res);

  return res;
}

char *
askforstring(const char *question,
	     const char *envname,
	     char *res,
	     size_t size)
{
  char buf[80];
  char *env;
  int i;

  env = getenv(envname);
  if(env && strlen(env) <= size) {
    strncpy(res, env, size);
    return res;
  }

  if(size > 80)
    size = 80;

  strncpy(buf, res, size);
  
  (void) printf("%s (\"%s\")\n", question, res);
  if(fgets(buf, 80, stdin) && buf[0] != '\n') {
    for(i=0; i<size && buf[i] != '\0'; i++)
      if(buf[i] == '\n')
	buf[i] = '\0';
    
    strncpy(res, buf, size);
  }

  return res;
}

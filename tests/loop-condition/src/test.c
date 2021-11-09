#include <stdio.h>

#ifndef ANGELIX_OUTPUT
#define ANGELIX_OUTPUT(type, expr, id) expr
#endif

int main(int argc, char *argv[]) {
  int n;
  int i = 10;
  int j = 20;
  n = atoi(argv[1]);

  if (n > 1) {
    n++;
    n--;
  }

  for (j = 0; j < 10; j++) {
    i--;
  }

  for (j = 0; j < 10; j++) {
    i++;
  }

  for (j = 0; j < 0; j++) {
    i--;
  }

  while (n > 1) { // n >= 1
    n--;
    printf("%d\n", ANGELIX_OUTPUT(int, n, "n"));
  }
  return 0;
}

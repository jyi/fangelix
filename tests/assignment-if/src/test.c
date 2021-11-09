#include <stdio.h>

#ifndef ANGELIX_OUTPUT
#define ANGELIX_OUTPUT(type, expr, id) expr
#endif

int main(int argc, char *argv[]) {
  int a, b, c, i;
  a = atoi(argv[1]);
  b = atoi(argv[2]);

  a = a + 0;
  c = a + b; // a - b

  if (b > c + 1) {
    c = c + 1;
    c = c - 1;
  }

  if (b > c - 1) {
    c = c + 1;
    c = c - 1;
  }  
  
  if (b > c) {
    fprintf(stderr, "[test.c] then\n");
    printf("%d\n", ANGELIX_OUTPUT(int, c + 1, "stdout"));
  } else {
    fprintf(stderr, "[test.c] else\n");    
    printf("%d\n", ANGELIX_OUTPUT(int, c - 1, "stdout"));    
  }
  return 0;
}

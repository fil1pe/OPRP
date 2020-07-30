#ifndef MATRIZ_OP_PTHREAD
#define MATRIZ_OP_PTHREAD

#include "matrizv3.h"

void *th_func1(void*);
mymatriz *mmultiplicar_openmp(mymatriz* matA, mymatriz* matB, int nth);
void *th_func2(void*);
mymatriz *mmultiplicar_openmp_blocos(mymatriz* matA, mymatriz* matB, int nth);

#endif
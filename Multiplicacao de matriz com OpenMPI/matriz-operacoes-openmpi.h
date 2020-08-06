#ifndef MATRIZ_OP_PTHREAD
#define MATRIZ_OP_PTHREAD

#include "matrizv3.h"

void *th_func1(void*);
mymatriz *mmultiplicar_openmpi(mymatriz* matA, mymatriz* matB, int rank, int size);
void *th_func2(void*);
mymatriz *mmultiplicar_openmpi_blocos(mymatriz* matA, mymatriz* matB, int rank, int size);

#endif
#ifndef MATRIZ_OP_PTHREAD
#define MATRIZ_OP_PTHREAD

#include "matrizv3.h"

typedef struct {
    mymatriz *A;
    mymatriz *B;
    mymatriz *C;
    int linha;
    int nthreads;
} mat_mult_th;

typedef struct {
    matriz_bloco_t *A;
    matriz_bloco_t *B;
    matriz_bloco_t *C;
} block_mat_mult_th;

void *th_func1(void*);
mymatriz *mmultiplicar_pthread(mymatriz* matA, mymatriz* matB, int nth);
void *th_func2(void*);
mymatriz *mmultiplicar_pthread_blocos(mymatriz* matA, mymatriz* matB, int nth);

#endif
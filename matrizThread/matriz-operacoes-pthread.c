#include "matriz-operacoes-pthread.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "matrizv3.h"
#include "matriz-operacoesv3.h"
#include "matriz-structs.h"

void *th_func1(void *p) {
    mat_mult_th *pc = (mat_mult_th*) p;

    for (int i=pc->linha; i<pc->A->lin; i+=pc->nthreads)
        for (int j=0; j<pc->B->col; j++)
            for(int k=0; k<pc->A->col; k++)
                pc->C->matriz[i][j] += pc->A->matriz[i][k] * pc->B->matriz[k][j];
    
    return NULL;
}

mymatriz *mmultiplicar_pthread(mymatriz* matA, mymatriz* matB, int nth) {
    mat_mult_th *p = (mat_mult_th*) malloc (nth * sizeof(mat_mult_th));
    pthread_t *threads = (pthread_t *) malloc (nth * sizeof(pthread_t));

    mymatriz *matC = (mymatriz*) malloc(sizeof(mymatriz));
    matC->lin = matA->lin;
    matC->col = matB->col;
    malocar(matC);
    mzerar(matC);

    int i;

    for (i=0; i<nth; i++) {
        p[i].A = matA;
        p[i].B = matB;
        p[i].C = matC;
        p[i].nthreads = nth;
        p[i].linha = i;
        pthread_create(&threads[i], NULL, th_func1, (void *) (p + i));
    }

    for (i=0; i<nth; i++)
        pthread_join(threads[i], NULL);
    
    free(p);
    free(threads);
    
    return matC;
}

void *th_func2(void *p) {
    block_mat_mult_th *pc = (block_mat_mult_th*) p;
    multiplicar_submatriz(pc->A, pc->B, pc->C);
    return NULL;
}

mymatriz *mmultiplicar_pthread_blocos(mymatriz* matA, mymatriz* matB, int nth) {
    block_mat_mult_th *p = (block_mat_mult_th*) malloc (nth * sizeof(block_mat_mult_th));
    pthread_t *threads = (pthread_t *) malloc (nth * sizeof(pthread_t));

    matriz_bloco_t **A = particionar_matriz(matA->matriz, matA->lin, matA->col, 1, nth);
    matriz_bloco_t **B = particionar_matriz(matB->matriz, matB->lin, matB->col, 0, nth);
    matriz_bloco_t **C = constroi_submatrizv2(matA->lin, matB->col, nth);

    int i;

    for (i=0; i<nth; i++) {
        p[i].A = A[i];
        p[i].B = B[i];
        p[i].C = C[i];
        pthread_create(&threads[i], NULL, th_func2, (void *) (p + i));
    }

    mymatriz *res = (mymatriz*)malloc(sizeof(mymatriz));
    res->lin = matA->lin;
    res->col = matB->col;
    malocar(res);
    mzerar(res);

    for (i=0; i<nth; i++) {
        pthread_join(threads[i], NULL);
        res = msomar(res, C[i]->matriz, 0);
        free(A[i]->bloco);
        free(A[i]);
        free(B[i]->bloco);
        free(B[i]);
        free(C[i]->bloco);
        mliberar(C[i]->matriz);
        free(C[i]->matriz);
        free(C[i]);
    }

    free(A);
    free(B);
    free(C);
    free(p);
    free(threads);

    return res;
}
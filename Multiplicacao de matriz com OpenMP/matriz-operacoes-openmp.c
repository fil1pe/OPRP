#include "matriz-operacoes-openmp.h"

#include <omp.h>
#include <stdlib.h>

#include "matrizv3.h"
#include "matriz-operacoesv3.h"
#include "matriz-structs.h"

mymatriz *mmultiplicar_openmp(mymatriz* matA, mymatriz* matB, int nth) {
    mymatriz *matC = (mymatriz*) malloc(sizeof(mymatriz));
    matC->lin = matA->lin;
    matC->col = matB->col;
    malocar(matC);
    mzerar(matC);

    int i, j, k;

    #pragma omp parallel for num_threads(nth) private(i, j, k)
    for (i=0; i<matA->lin; i++)
        for (j=0; j<matB->col; j++)
            for(k=0; k<matA->col; k++)
                matC->matriz[i][j] += matA->matriz[i][k] * matB->matriz[k][j];

    return matC;
}

mymatriz *mmultiplicar_openmp_blocos(mymatriz* matA, mymatriz* matB, int nth) {
    matriz_bloco_t **A = particionar_matriz(matA->matriz, matA->lin, matA->col, 1, nth);
    matriz_bloco_t **B = particionar_matriz(matB->matriz, matB->lin, matB->col, 0, nth);
    matriz_bloco_t **C = constroi_submatrizv2(matA->lin, matB->col, nth);

    mymatriz *res = (mymatriz*)malloc(sizeof(mymatriz));
    res->lin = matA->lin;
    res->col = matB->col;
    malocar(res);
    mzerar(res);

    int tid;

    #pragma omp parallel num_threads(nth) private(tid)
    {

    tid = omp_get_thread_num();

    multiplicar_submatriz(A[tid], B[tid], C[tid]);

    #pragma omp critical
    {
    mymatriz *aux = msomar(res, C[tid]->matriz, 0);
    mliberar(res);
    free(res);
    res = aux;
    }

    free(A[tid]->bloco);
    free(A[tid]);
    free(B[tid]->bloco);
    free(B[tid]);
    free(C[tid]->bloco);
    mliberar(C[tid]->matriz);
    free(C[tid]->matriz);
    free(C[tid]);

    }

    free(A);
    free(B);
    free(C);

    return res;
}
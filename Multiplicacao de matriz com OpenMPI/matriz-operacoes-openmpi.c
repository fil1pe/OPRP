#include <mpi.h>

#include "matriz-operacoes-openmpi.h"

#include <stdlib.h>
#include <math.h>

#include "matrizv3.h"
#include "matriz-operacoesv3.h"
#include "matriz-structs.h"

mymatriz *mmultiplicar_openmpi(mymatriz* mat_a, mymatriz* mat_b, int rank, int size) {
    mymatriz *mat_c = NULL;

    // Envia a matriz B completa para todo mundo:
    MPI_Bcast(&mat_b->lin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mat_b->col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank)
        malocar(mat_b);

    MPI_Bcast(&mat_b->matriz[0][0], (mat_b->lin) * (mat_b->col), MPI_INT, 0, MPI_COMM_WORLD);

    // Envia o número de linhas de A:
    int a_lin;
    if (!rank)
        a_lin = mat_a->lin;
    MPI_Bcast(&a_lin, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Faz scatter das linhas da matriz A:
    mymatriz *mat_a_block = (mymatriz*) malloc(sizeof(mymatriz));
    mat_a_block->lin = ceil(a_lin/(float)size);
    mat_a_block->col = mat_b->lin;
    malocar(mat_a_block);

    if (!rank)
        MPI_Scatter(&mat_a->matriz[0][0], mat_a_block->lin * mat_a_block->col, MPI_INT, &mat_a_block->matriz[0][0], mat_a_block->lin * mat_a_block->col, MPI_INT, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, mat_a_block->lin * mat_a_block->col, MPI_INT, &mat_a_block->matriz[0][0], mat_a_block->lin * mat_a_block->col, MPI_INT, 0, MPI_COMM_WORLD);

    // Gera matriz bloco para compor C:
    mymatriz *mat_c_block = mmultiplicar(mat_a_block, mat_b, 0);

    // Une os blocos em C:
    if (!rank) {
        mat_c = (mymatriz *) malloc (sizeof(mymatriz));
        mat_c->lin = size * mat_a_block->lin;
        mat_c->col = mat_b->col;
        malocar(mat_c);

        MPI_Gather(&mat_c_block->matriz[0][0], mat_c_block->lin * mat_c_block->col, MPI_INT, &mat_c->matriz[0][0], mat_c_block->lin * mat_c_block->col, MPI_INT, 0, MPI_COMM_WORLD);

        mat_c->lin = mat_a->lin;
    } else
        MPI_Gather(&mat_c_block->matriz[0][0], mat_c_block->lin * mat_c_block->col, MPI_INT, NULL, mat_c_block->lin * mat_c_block->col, MPI_INT, 0, MPI_COMM_WORLD);

    // Libera a memória dos blocos:
    mliberar(mat_c_block);
    free(mat_c_block);
    mliberar(mat_a_block);
    free(mat_a_block);

    return mat_c;
}

mymatriz *mmultiplicar_openmpi_blocos(mymatriz* matA, mymatriz* matB, int nth) {
    /*matriz_bloco_t **A = particionar_matriz(matA->matriz, matA->lin, matA->col, 1, nth);
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

    return res;*/
    return matA;
}
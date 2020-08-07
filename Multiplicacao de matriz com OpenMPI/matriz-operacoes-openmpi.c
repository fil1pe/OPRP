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

    MPI_Bcast(&mat_b->matriz[0][0], mat_b->lin * mat_b->col, MPI_INT, 0, MPI_COMM_WORLD);

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

mymatriz *mmultiplicar_openmpi_blocos(mymatriz* mat_a, mymatriz* mat_b, int rank, int size) {
    int a_lin, a_col, b_lin, b_col;

    // Envia os dados das matrizes:
    if (!rank) {
        a_lin = mat_a->lin;
        b_lin = mat_b->lin;
        b_col = mat_b->col;
    }
    MPI_Bcast(&a_lin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b_lin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    a_col = b_lin;
    MPI_Bcast(&b_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Define os cortes para os blocos:
    int divisor = ceil(a_col/(float)size);
    int start = divisor * rank;
    if (rank == size - 1)
        divisor = a_col - rank * divisor;

    // Envia as matrizes A e B:
    if (rank) {
        mat_a->lin = a_lin;
        mat_a->col = a_col;
        mat_b->lin = b_lin;
        mat_b->col = b_col;
        malocar(mat_a);
        malocar(mat_b);
    }
    MPI_Bcast(&mat_a->matriz[0][0], a_lin * a_col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mat_b->matriz[0][0], b_lin * b_col, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Define os blocos:
    matriz_bloco_t *bloco_a = particionar_matrizv2(mat_a, 1, divisor, start);
    matriz_bloco_t *bloco_b = particionar_matrizv2(mat_b, 0, divisor, start);
    matriz_bloco_t *bloco_c = constroi_submatrizv3(a_lin, b_col);

    // Multiplica os blocos:
    multiplicar_submatriz(bloco_a, bloco_b, bloco_c);

    // Envia os blocos para compor a matriz C=A*B:
    MPI_Status status;
    if (rank)
        MPI_Send(&bloco_c->matriz->matriz[0][0], a_lin * b_col, MPI_INT, 0, 0, MPI_COMM_WORLD);
    else {
        matriz_bloco_t *aux0 = constroi_submatrizv3(a_lin, b_col);
        for (int i=1; i<size; i++) {
            MPI_Recv(&aux0->matriz->matriz[0][0], a_lin * b_col, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            mymatriz *aux1 = msomar(bloco_c->matriz, aux0->matriz, 0);
            mliberar(bloco_c->matriz);
            free(bloco_c->matriz);
            bloco_c->matriz = aux1;
        }
        free(aux0->bloco);
        mliberar(aux0->matriz);
        free(aux0->matriz);
        free(aux0);
    }

    // Libera espaço na memória:
    free(bloco_a->bloco);
    free(bloco_a);
    free(bloco_b->bloco);
    free(bloco_b);
    free(bloco_c->bloco);
    if (rank) {
        mliberar(bloco_c->matriz);
        free(bloco_c->matriz);
    }
    mymatriz *res = bloco_c->matriz;
    free(bloco_c);

    return res;
}
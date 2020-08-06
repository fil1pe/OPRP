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

mymatriz *mmultiplicar_openmpi_blocos(mymatriz* matA, mymatriz* matB, int rank, int size) {
    mymatriz *bloco_A, *bloco_B;
    int lin_col[2];

    if (!rank) {
        // Obtém a transposta de A:
        mymatriz *matA_trans = transposta(matA);

        // Gera os blocos:
        mymatriz **A = particionar_matrizv2(matA_trans, size);
        mymatriz **B = particionar_matrizv2(matB, size);

        // Envia os blocos:
        bloco_A = A[0];
        bloco_B = B[0];

        for (int i=1; i<size; i++) {
            lin_col[0] = A[i]->lin;
            lin_col[1] = A[i]->col;
            MPI_Send(lin_col, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&A[i]->matriz[0][0], lin_col[0] * lin_col[1], MPI_INT, i, 0, MPI_COMM_WORLD);

            lin_col[0] = B[i]->lin;
            lin_col[1] = B[i]->col;
            MPI_Send(lin_col, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&B[i]->matriz[0][0], lin_col[0] * lin_col[1], MPI_INT, i, 0, MPI_COMM_WORLD);

            mliberar(A[i]);
            free(A[i]);
            mliberar(B[i]);
            free(B[i]);
        }

        // Libera memória:
        mliberar(matA_trans);
        free(matA_trans);
        free(A);
        free(B);
    } else {
        MPI_Status status;
        MPI_Recv(lin_col, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        bloco_A = (mymatriz*) malloc(sizeof(mymatriz));
        bloco_A->lin = lin_col[0];
        bloco_A->col = lin_col[1];
        malocar(bloco_A);
        MPI_Recv(&bloco_A->matriz[0][0], lin_col[0] * lin_col[1], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        MPI_Recv(lin_col, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        bloco_B = (mymatriz*) malloc(sizeof(mymatriz));
        bloco_B->lin = lin_col[0];
        bloco_B->col = lin_col[1];
        malocar(bloco_B);
        MPI_Recv(&bloco_B->matriz[0][0], lin_col[0] * lin_col[1], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    bloco_A = transposta(bloco_A);

    mymatriz *C = mmultiplicar(bloco_A, bloco_B, 0);

    if (rank) {
        lin_col[0] = C->lin;
        lin_col[1] = C->col;
        MPI_Send(lin_col, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&C->matriz[0][0], lin_col[0] * lin_col[1], MPI_INT, 0, 0, MPI_COMM_WORLD);
        mliberar(C);
        free(C);
    } else {
        for (int i=1; i<size; i++) {
            MPI_Status status;
            MPI_Recv(lin_col, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            mymatriz *bloco_C = (mymatriz*) malloc(sizeof(mymatriz));
            bloco_C->lin = lin_col[0];
            bloco_C->col = lin_col[1];
            malocar(bloco_C);
            MPI_Recv(&bloco_C->matriz[0][0], lin_col[0] * lin_col[1], MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            C = msomar(C, bloco_C, 0);
        }
    }

    mliberar(bloco_A);
    free(bloco_A);
    mliberar(bloco_B);
    free(bloco_B);

    return C;
}
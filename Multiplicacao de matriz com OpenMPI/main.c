#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matriz-structs.h"
#include "toolsv3.h"
#include "matrizv3.h"
#include "matriz-operacoesv3.h"
#include "matriz-operacoes-openmpi.h"
#define EXECUTIONS 4

int main(int argc, char *argv[]) {
    int rank, nthreads;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nthreads);

    mymatriz *mat_a, *mat_b, **mmultbloco, **mmult, **thread_mmultbloco, **thread_mmult;
    char filename[100];
    FILE *fmat;
    int nr_line;
    int *vet_line;
    int N, M, La, Lb;
    double start_time, end_time;
    double seq_normal_times[EXECUTIONS];
    double thr_normal_times[EXECUTIONS];
    double seq_bloco_times[EXECUTIONS];
    double thr_bloco_times[EXECUTIONS];
    matriz_bloco_t **submatA, **submatB, **submatC;
    int i, j;

    mat_b = (mymatriz*) malloc(sizeof(mymatriz));

    if (!rank) {
        mat_a = (mymatriz*) malloc(sizeof(mymatriz));
        mmultbloco = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
        mmult = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
        thread_mmultbloco = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
        thread_mmult = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));

        if (argc < 3) {
            printf("ERRO: número de parametros %s <matriz_a> <matriz_b>\n", argv[0]);
            exit(1);
        }


        fmat = fopen(argv[1], "r");
        if (fmat == NULL) {
            printf("ERRO na abertura dos arquivos\n");
            exit(1);
        }
        extrai_parametros_matriz(fmat, &N, &La, &vet_line, &nr_line);
        
        mat_a->matriz = NULL;
        mat_a->lin = N;
        mat_a->col = La;
        if (malocar(mat_a)) {
            printf("ERROR: Out of memory\n");
        }
        filein_matriz(mat_a->matriz, N, La, fmat, vet_line, nr_line);
        free(vet_line);
        fclose(fmat);
        

        fmat = fopen(argv[2], "r");
        if (fmat == NULL) {
            printf("ERRO na abertura dos arquivos\n");
            exit(1);
        }
        extrai_parametros_matriz(fmat, &Lb, &M, &vet_line, &nr_line);
        mat_b->matriz = NULL;
        mat_b->lin = Lb;
        mat_b->col = M;
        if (malocar(mat_b)) {
            printf("ERROR: Out of memory\n");
        }
        filein_matriz(mat_b->matriz, Lb, M, fmat, vet_line, nr_line);
        free(vet_line);
        fclose(fmat);


        i = EXECUTIONS;
        while (i--) {
            j = EXECUTIONS - 1 - i;
            printf("##### Multiplicação normal (mult%d) #####\n", j);
            start_time = wtime();
            mmult[j] = mmultiplicar(mat_a, mat_b, 1);
            end_time = wtime();

            if (!mmult[j]) {
                exit(1);
            }

            printf("  Tempo: %f s\n", end_time - start_time);
            sprintf(filename, "mult%d.result", j);
            fmat = fopen(filename, "w");
            fileout_matriz(mmult[j], fmat);
            fclose(fmat);

            seq_normal_times[j] = end_time - start_time;
        }

        printf("\n");
    }

    i = EXECUTIONS;
    while (i--) {
        if (!rank) {
            j = EXECUTIONS - 1 - i;
            printf("##### Multiplicação normal com OpenMPI (mult_openmpi%d) #####\n", j);
            start_time = wtime();

            thread_mmult[j] = mmultiplicar_openmpi(mat_a, mat_b, rank, nthreads);
        } else
            mmultiplicar_openmpi(mat_a, mat_b, rank, nthreads);

        if (!rank) {
            end_time = wtime();

            printf("  Tempo: %f s\n", end_time - start_time);
            sprintf(filename, "mult_openmpi%d.result", j);
            fmat = fopen(filename, "w");
            fileout_matriz(thread_mmult[j], fmat);
            fclose(fmat);

            thr_normal_times[j] = (end_time - start_time);
        }
    }
    
    if (!rank) {
        printf("\n");
        i = EXECUTIONS;
        while (i--) {
            j = EXECUTIONS - 1 - i;
            printf("##### Multiplicação em bloco (mult_block%d) #####\n", j);
            start_time = wtime();

            submatA = particionar_matriz(mat_a->matriz, N, La, 1, 2);
            submatB = particionar_matriz(mat_b->matriz, Lb, M, 0, 2);
            submatC = constroi_submatrizv2(N, M, 2);

            multiplicar_submatriz(submatA[0], submatB[0], submatC[0]);
            multiplicar_submatriz(submatA[1], submatB[1], submatC[1]);

            mmultbloco[j] = msomar(submatC[0]->matriz, submatC[1]->matriz, 0);

            end_time = wtime();

            if (!mmultbloco[j]) {
                exit(1);
            }
            printf("  Tempo: %f s\n", end_time - start_time);
            sprintf(filename, "mult_block%d.result", j);
            fmat = fopen(filename, "w");
            fileout_matriz(mmultbloco[j], fmat);
            fclose(fmat);

            seq_bloco_times[j] = end_time - start_time;
        }

        printf("\n");
    }

    i = EXECUTIONS;
    while (i--) {
        if (!rank) {
            j = EXECUTIONS - 1 - i;
            printf("##### Multiplicação em bloco com OpenMPI (mult_block_openmpi%d) #####\n", j);
            start_time = wtime();

            thread_mmultbloco[j] = mmultiplicar_openmpi_blocos(mat_a, mat_b, rank, nthreads);
        } else
            mmultiplicar_openmpi_blocos(mat_a, mat_b, rank, nthreads);

        if (!rank) {
            end_time = wtime();
            
            printf("  Tempo: %f s\n", end_time - start_time);
            sprintf(filename, "mult_block_openmpi%d.result", j);
            fmat = fopen(filename, "w");
            fileout_matriz(thread_mmultbloco[j], fmat);
            fclose(fmat);

            thr_bloco_times[j] = end_time - start_time;
        }
    }
    
    if (!rank) {
        printf("\n##### Comparação dos resultados da multiplicação #####\n");

        for (int i = 0; i < EXECUTIONS; i++) {
            printf("mult%d vs. mult_openmpi%d\n ˪", i, i);
            mcomparar(mmult[i], thread_mmult[i]);
        }

        for (int i = 0; i < EXECUTIONS; i++) {
            printf("mult%d vs. mult_block%d\n ˪", i, i);
            mcomparar(mmult[i], mmultbloco[i]);
        }

        for (int i = 0; i < EXECUTIONS; i++) {
            printf("mult%d vs. mult_block_openmpi%d\n ˪", i, i);
            mcomparar(mmult[i], thread_mmultbloco[i]);
        }

        printf("\n##### Tempos médios (%d execuções) #####\n", EXECUTIONS);
        double tempo_normal_seq, tempo_normal_thr, tempo_bloco_seq, tempo_bloco_thr;

        tempo_normal_seq = tempo_normal_thr = tempo_bloco_seq = tempo_bloco_thr = 0;

        for (int i = 0; i < EXECUTIONS; i++) {
            tempo_normal_seq += seq_normal_times[i];
            tempo_normal_thr += thr_normal_times[i];
            tempo_bloco_seq += seq_bloco_times[i];
            tempo_bloco_thr += thr_bloco_times[i];
        }

        double media_normal_seq = tempo_normal_seq / (float)EXECUTIONS,
            media_normal_thr = tempo_normal_thr / (float)EXECUTIONS,
            media_bloco_seq = tempo_bloco_seq / (float)EXECUTIONS,
            media_bloco_thr = tempo_bloco_thr / (float)EXECUTIONS;

        printf(
            "Multiplicação normal sequencial:\t\t\t%02lf s\
            \nMultiplicação normal com \033[1m%d processos\033[0m (OpenMPI):\t\t%02lf s\
            \nMultiplicação em bloco sequencial:\t\t\t%02lf s\
            \nMultiplicação em bloco com \033[1m%d processos\033[0m (OpenMPI):\t%02lf s\n",
            media_normal_seq,
            nthreads,
            media_normal_thr,
            media_bloco_seq,
            nthreads,
            media_bloco_thr
        );

        double speedup_normal = media_normal_seq / media_normal_thr;
        double speedup_bloco = media_bloco_seq / media_bloco_thr;
        printf("\n##### Speedups #####\n");
        printf("Multiplicação normal:\t%03lf\nMultiplicação bloco:\t%03lf\n", speedup_normal, speedup_bloco);

        mliberar(mmult[0]);
        free(mmult[0]);
        mliberar(mmultbloco[0]);
        free(mmultbloco[0]);
        mliberar(mat_a);
        free(mat_a);
        free(mmult);
        free(mmultbloco);
    }

    mliberar(mat_b);
    free(mat_b);

    MPI_Finalize();
    return 0;
}
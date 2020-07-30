#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matriz-structs.h"
#include "toolsv3.h"
#include "matrizv3.h"
#include "matriz-operacoesv3.h"
#include "matriz-operacoes-pthread.h"
#define EXECUTIONS 4

int main(int argc, char *argv[]) {
    mymatriz *mat_a = (mymatriz*) malloc(sizeof(mymatriz));
    mymatriz *mat_b = (mymatriz*) malloc(sizeof(mymatriz));
    mymatriz **mmultbloco = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
    mymatriz **mmult = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
    mymatriz **thread_mmultbloco = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
    mymatriz **thread_mmult = (mymatriz **) malloc(EXECUTIONS * sizeof(mymatriz *));
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
    int i, j;
    matriz_bloco_t **submatA, **submatB, **submatC;


    if (argc != 4) {
        printf("ERRO: número de parametros %s <matriz_a> <matriz_b> <nthreads>\n", argv[0]);
        exit(1);
    }
    int nthreads = atoi(argv[3]);
    if (nthreads <= 0) {
        printf("ERRO: número inválido de threads\n");
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
        printf("##### Multiplicação simples %d #####\n", j);
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
    i = EXECUTIONS;
    while (i--) {
        j = EXECUTIONS - 1 - i;
        printf("##### THREAD - Multiplicação simples %d #####\n", j);
        start_time = wtime();
        thread_mmult[j] = mmultiplicar_pthread(mat_a, mat_b, nthreads);
        end_time = wtime();

        if (!thread_mmult[j]) {
            exit(1);
        }

        printf("  Tempo: %f s\n", end_time - start_time);
        sprintf(filename, "mult_thread%d.result", j);
        fmat = fopen(filename, "w");
        fileout_matriz(thread_mmult[j], fmat);
        fclose(fmat);

        thr_normal_times[j] = (end_time - start_time);
    }
    
    
    printf("\n");
    i = EXECUTIONS;
    while (i--) {
        j = EXECUTIONS - 1 - i;
        printf("##### Multiplicação em bloco %d #####\n", j);
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
    i = EXECUTIONS;
    while (i--) {
        j = EXECUTIONS - 1 - i;
        printf("##### THREAD - Multiplicação em bloco %d #####\n", j);
        start_time = wtime();

        thread_mmultbloco[j] = mmultiplicar_pthread_blocos(mat_a, mat_b, nthreads);

        end_time = wtime();

        if (!thread_mmultbloco[j]) {
            exit(1);
        }
        printf("  Tempo: %f s\n", end_time - start_time);
        sprintf(filename, "mult_block_thread%d.result", j);
        fmat = fopen(filename, "w");
        fileout_matriz(thread_mmultbloco[j], fmat);
        fclose(fmat);

        thr_bloco_times[j] = end_time - start_time;
    }
    
    
    printf("\n##### Comparação dos resultados da multiplicação #####\n");

    for (int i = 0; i < EXECUTIONS; i++) {
        printf("[mult%d vs mult_thread%d]\t\t", i, i);
        mcomparar(mmult[i], thread_mmult[i]);
    }

    for (int i = 0; i < EXECUTIONS; i++) {
        printf("[mult%d vs mult_block%d]\t\t", i, i);
        mcomparar(mmult[i], mmultbloco[i]);
    }

    for (int i = 0; i < EXECUTIONS; i++) {
        printf("[mult%d vs mult_block_thread%d]\t", i, i);
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
        "Multiplicação simples sequencial:\t%02lf s\
        \nMultiplicação simples com %d threads:\t%02lf s\
        \nMultiplicação em bloco sequencial:\t%02lf s\
        \nMultiplicação em bloco com %d threads:\t%02lf s\n",
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
    mliberar(mat_b);
    free(mat_b);
    free(mmult);
    free(mmultbloco);


    return 0;
}
#ifndef MATRIZ_STRUCT_H
#define MATRIZ_STRUCT_H

typedef struct {
    int **matriz;
    int lin;
    int col;
} mymatriz;

typedef struct {
    int lin_inicio;
    int lin_fim;
    int col_inicio;
    int col_fim;
} bloco_t;

typedef struct {
    mymatriz *matriz;
    bloco_t *bloco;
} matriz_bloco_t;

#endif
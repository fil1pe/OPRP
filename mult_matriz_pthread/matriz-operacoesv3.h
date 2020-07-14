#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "matriz-structs.h"

mymatriz *msomar(mymatriz *mat_a, mymatriz *mat_b, int tipo);
mymatriz *mmultiplicar(mymatriz *mat_a, mymatriz *mat_b, int tipo);

int multiplicar_submatriz(matriz_bloco_t *mat_suba, matriz_bloco_t *mat_subb, matriz_bloco_t *mat_subc);
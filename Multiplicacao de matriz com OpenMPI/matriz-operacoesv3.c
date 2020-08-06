#include "matriz-operacoesv3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "matrizv3.h"

mymatriz *msomar(mymatriz *mat_a, mymatriz *mat_b, int tipo) {
  mymatriz *mat_c = NULL;

	if ((mat_a->lin != mat_b-> lin) || (mat_a->col != mat_b->col)){
		//printf ("Erro: Matrizes incompatíveis!\n");
		//exit(1);
    return mat_a;
	}

	mat_c = (mymatriz *) malloc (sizeof(mymatriz));
	mat_c->lin = mat_a->lin;
	mat_c->col = mat_a->col;

	if (malocar(mat_c)) printf ("ERROR: Out of memory\n");

  if (tipo == 1) {
		for (int i=0; i < mat_c->lin; i++)
		  for (int j=0; j < mat_c->col; j++)
					mat_c->matriz[i][j] = mat_a->matriz[i][j]+mat_b->matriz[i][j];
	} else {
		for (int j=0; j < mat_c->col; j++)
			for (int i=0; i < mat_c->lin; i++)
					mat_c->matriz[i][j] = mat_a->matriz[i][j]+mat_b->matriz[i][j];
	}

  return mat_c;
}

mymatriz *mmultiplicar(mymatriz *mat_a, mymatriz *mat_b, int tipo) {
  mymatriz *mat_c = NULL;

  if (mat_a->col != mat_b->lin){
		printf ("Erro: Matrizes incompatíveis!\n");
		exit(1);
	}

  mat_c = (mymatriz *) malloc (sizeof(mymatriz));
	mat_c->lin = mat_a->lin;
	mat_c->col = mat_b->col;

  if (malocar(mat_c)) printf ("ERROR: Out of memory\n");

  mzerar(mat_c);

  if (tipo == 1) {
    for (int i=0; i<mat_c->lin; i++)
      for (int j=0; j<mat_c->col; j++)
        for (int k=0; k<mat_a->col; k++)
          mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
  } else if (tipo == 2) {
    for (int j=0; j<mat_c->col; j++)
      for (int i=0; i<mat_c->lin; i++)
        for (int k=0; k<mat_a->col; k++)
          mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
  } else if (tipo == 3) {
    for (int i=0; i<mat_c->lin; i++)
      for (int k=0; k<mat_a->col; k++)
        for (int j=0; j<mat_c->col; j++)
          mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
  } else if (tipo == 4) {
    for (int k=0; k<mat_a->col; k++)
      for (int i=0; i<mat_c->lin; i++)
        for (int j=0; j<mat_c->col; j++)
          mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
  } else if (tipo == 5) {
    for (int k=0; k<mat_a->col; k++)
      for (int j=0; j<mat_c->col; j++)
        for (int i=0; i<mat_c->lin; i++)
          mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
  } else {
    for (int j=0; j<mat_c->col; j++)
      for (int k=0; k<mat_a->col; k++)
        for (int i=0; i<mat_c->lin; i++)
          mat_c->matriz[i][j] += mat_a->matriz[i][k] * mat_b->matriz[k][j];
  }

  return mat_c;
}

mymatriz *transposta(mymatriz *mat) {
  mymatriz *mat_trans = (mymatriz *) malloc (sizeof(mymatriz));
	mat_trans->lin = mat->col;
	mat_trans->col = mat->lin;

  if (malocar(mat_trans)) printf ("ERROR: Out of memory\n");

  for (int i=0; i<mat_trans->lin; i++)
    for (int j=0; j<mat_trans->col; j++)
      mat_trans->matriz[i][j] = mat->matriz[j][i];
  
  return mat_trans;
}

int multiplicar_submatriz(matriz_bloco_t *mat_suba, matriz_bloco_t *mat_subb, matriz_bloco_t *mat_subc) {
  for (int i = mat_suba->bloco->lin_inicio; i<mat_suba->bloco->lin_fim; i++)
    for (int j = mat_subb->bloco->col_inicio; j<mat_subb->bloco->col_fim; j++) {
      mat_subc->matriz->matriz[i][j] = 0;
      for (int k = mat_suba->bloco->col_inicio; k<mat_suba->bloco->col_fim; k++)
        mat_subc->matriz->matriz[i][j] += mat_suba->matriz->matriz[i][k] * mat_subb->matriz->matriz[k][j];
    }

  return 1;
}
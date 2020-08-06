#include "matrizv3.h"
#include <time.h>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int malocar (mymatriz *matriz) {
	int **newMatriz = NULL;

	newMatriz = (int **) malloc(matriz->lin*sizeof(int *));

	if (!newMatriz) {
		printf("ERROR: Out of memory\n");
		return 1;
	}

	newMatriz[0] = (int *) malloc(matriz->lin * matriz->col * sizeof(int));
  	for (int i=1; i < matriz->lin; i++) {
		  	newMatriz[i] = newMatriz[0] + i * matriz->col;
			//newMatriz[i] = (int *) malloc(sizeof(int)*matriz->col);
			if (!newMatriz) {
				printf("ERROR: Out of memory\n");
				return 1;
			}
	}

	matriz->matriz = newMatriz;
	return 0;
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MATRIZ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int mgerar(mymatriz *matriz, int valor){
	srand( (unsigned)time(NULL) );

	for (int i=0; i < matriz->lin; i++)
	  for (int j=0; j < matriz->col; j++)
			if (valor == -9999)
				matriz->matriz[i][j] = rand() % 10;
			else
				matriz->matriz[i][j] = valor;

	return 0;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int mimprimir (mymatriz *matriz){
	int linha, coluna;
	linha = matriz->lin;
	coluna = matriz->col;

	if (linha > 15) {
		linha = 15;
	}

	if (coluna > 15) {
		coluna = 15;
	}

	for (int j =0; j < coluna; j++)
		printf("\t(%d)", j);
	printf("\n");
	for (int i=0; i < linha; i++) {
		printf("(%d)", i);
	  for (int j=0; j < coluna; j++){
			printf("\t%d", matriz->matriz[i][j]);
		}
		printf("\n");
	}

	printf("\n \
%%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%%\n \
	WARNING: Impressão truncada em 15x15! \n \
	WARNING: Último elemento matriz[%d][%d] = %d \n \
%%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%% %%%%%%%%%%%%\n", \
matriz->lin-1, matriz->col-1, matriz->matriz[matriz->lin-1][matriz->col-1]);
	return 0;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int mliberar (mymatriz *matriz) {
	/*for (int i =0; i < matriz->lin; i++) {
		//printf("%p\n", matriz->matriz[i]);
		free(matriz->matriz[i]);
	}*/
	free(matriz->matriz[0]);
	free(matriz->matriz);
	return 0;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int mzerar (mymatriz *matriz){
	return mgerar(matriz,0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int mcomparar (mymatriz *mat_a, mymatriz *mat_b){
	for (int j =0; j < mat_a->col; j++)
	for (int i=0; i < mat_a->lin; i++) {
		for (int j=0; j < mat_a->col; j++){
			if (mat_a->matriz[i][j] != mat_b->matriz[i][j]) {
				printf("O elemento [%d,%d] é diferente nas matrizes analisadas!\n", i,j);
				return 1;
			}
		}
	}
	printf("VERIFICADO: Matrizes idênticas\n");
	return 0;
}

matriz_bloco_t **particionar_matriz(int **matriz, int mat_lin, int mat_col, int orientacao, int divisor) {
  matriz_bloco_t **m = (matriz_bloco_t**) malloc(divisor*sizeof(matriz_bloco_t*));

  mymatriz *matriz_st = (mymatriz*) malloc(sizeof(mymatriz));
  matriz_st->matriz = matriz;

  int k;
  switch (orientacao) {
    case 0:
      k = mat_lin/divisor;
      for (int i=0; i<divisor; i++) {
          m[i] = (matriz_bloco_t*) malloc(sizeof(matriz_bloco_t));
          m[i]->bloco = (bloco_t*) malloc(sizeof(bloco_t));
          m[i]->matriz = matriz_st;
          m[i]->bloco->lin_inicio = k*i;
          m[i]->bloco->lin_fim = m[i]->bloco->lin_inicio + k;
          m[i]->bloco->col_inicio = 0;
          m[i]->bloco->col_fim = mat_col;
      }
      m[divisor-1]->bloco->lin_fim = mat_lin;
      break;

    default:
      k = mat_col/divisor;
      for (int i=0; i<divisor; i++) {
          m[i] = (matriz_bloco_t*) malloc(sizeof(matriz_bloco_t));
          m[i]->bloco = (bloco_t*) malloc(sizeof(bloco_t));
          m[i]->matriz = matriz_st;
          m[i]->bloco->col_inicio = k*i;
          m[i]->bloco->col_fim = m[i]->bloco->col_inicio + k;
          m[i]->bloco->lin_inicio = 0;
          m[i]->bloco->lin_fim = mat_lin;
      }
      m[divisor-1]->bloco->col_fim = mat_col;
  }

  return m;
}

matriz_bloco_t **constroi_submatrizv2(int mat_lin, int mat_col, int divisor) {
  matriz_bloco_t **m = (matriz_bloco_t**) malloc(divisor * sizeof(matriz_bloco_t*));

  for (int i=0; i<divisor; i++)
  {
    m[i] = (matriz_bloco_t*) malloc(sizeof(matriz_bloco_t));
    m[i]->bloco = (bloco_t*) malloc(sizeof(bloco_t));
    
    mymatriz *aux = (mymatriz*) malloc(sizeof(mymatriz));
    aux->lin = mat_lin;
    aux->col = mat_col;
    malocar(aux);
    mzerar(aux);

    m[i]->matriz = aux;
    m[i]->bloco->lin_inicio = 0;
    m[i]->bloco->lin_fim = mat_lin;
    m[i]->bloco->col_inicio = 0;
    m[i]->bloco->col_fim = mat_col;
  }

  return m;
}
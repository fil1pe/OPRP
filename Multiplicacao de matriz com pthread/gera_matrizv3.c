#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toolsv3.h"
#include "matrizv3.h"


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int main(int argc, char **argv) {
	FILE *fmat0;
	FILE *fmat1;
	mymatriz mma0,mma1;
	int linha, coluna;
	char filename [100];
	int *vet_line = NULL;
	int nr_line;


	if (argc != 3){
		printf ("ERRO: Numero de parametros %s <nr_linhas> <nr_colunas>", argv[0]);
		exit (1);
	}

	linha =  atoi(argv[1]);
	coluna = atoi(argv[2]);
	sprintf (filename, "%dx%d-mat.map", linha, coluna);

	fmat0 = fopen(filename,"w");
  if (fmat0 == NULL) {
		printf("Error: Na abertura dos arquivos.");
		exit(1);
  }

	mma0.matriz = NULL;
	mma0.lin = linha;
	mma0.col = coluna;
	if (malocar(&mma0)) {	//
	// printf("##### Arquivo %dx%d-mat.map: VERIFICADO! #####\n", linha, coluna);

		printf ("ERROR: Out of memory\n");
	}
	mgerar(&mma0, -9999);
	mimprimir(&mma0);

	printf("\t\t**** PRINT mat_c NxM(%d,%d) **** \n", linha, coluna);
	fileout_matriz(&mma0, fmat0);
	mliberar (&mma0);
	printf("#####\n Arquivo com a matriz gerada (%dx%d-mat.map).\n#####\n", linha, coluna);
	fclose(fmat0);

	fmat1 = fopen(filename,"r");
	if (fmat1 == NULL) {
		printf("Error: Na abertura dos arquivos.");
		exit(1);
	}

	extrai_parametros_matriz(fmat1, &mma1.lin, &mma1.col, &vet_line, &nr_line);
	if (malocar(&mma1)) {
		printf ("ERROR: Out of memory\n");
	}

	filein_matriz(mma1.matriz, linha, coluna, fmat1, vet_line, nr_line);
	mimprimir(&mma1);
	mliberar(&mma1);
	free(vet_line);
  fclose(fmat1);

	printf("##### Arquivo %dx%d-mat.map: VERIFICADO! #####\n", linha, coluna);
  return 0;
}
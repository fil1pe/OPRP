#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matriz-structs.h"

// externos a biblioteca tools
double wtime();
int extrai_parametros_matriz(FILE *const fin, int *linha, int *coluna, int **vet_line, int *nr_line);

// Internos a biblioteca tools
char **split(const char *str, char sep, int nro_palavras);
int splitv2(const char *str, char sep, int nro_line, int *vetnum);
int fileout_matriz(mymatriz *matriz, FILE *infile);
int contabiliza_file(FILE *const fin, int *line_length);
char *get_line_file(FILE *const fin, int get_line, int nro_line, int *vet_line);
int filein_matriz(int **matriz, int linha, int coluna, FILE *file, int *vet_ind, int nr_line);
int conta_line_file(FILE *const fin, int get_line, int nro_line, int *vet_line);
int nr_line_file(FILE *const fin, int *vet_line);
#include "toolsv3.h"
#include <time.h>
#include <sys/time.h>

#define _DEBUG_ 0

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FERRAMENTAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + (double) t.tv_usec / 1000000;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int splitv2(const char *str, char sep, int nro_line, int *vetnum) {
  int conta_palavras;
  unsigned int start = 0, stop = 0;
  char *tmpStr;

  #if _DEBUG_
  printf("SPLITv2: \n\tConteúdo da linha (%s)\n\tSeparador (%c)\n\tNro de caracteres na Linha(%d)\n", str, sep, nro_line);
  #endif

  conta_palavras = 0;

  #if _DEBUG_
  printf("SPLITv2: \tstr[stop] = %p %d\n", str, stop);
  #endif

  for (stop = 0; str[stop]; stop++) {
    if (str[stop] == sep) {

      #if _DEBUG_
      printf("SPLITv2:\tconta_palavras (%d) nro_caracteres_na_linha (%d) stop (%d)\n", conta_palavras, nro_line, stop);
      #endif

      if ( conta_palavras < nro_line ) {

        #if _DEBUG_
        printf("SPLITv2:\t%d %d\n", conta_palavras, nro_line);
        #endif

        tmpStr = (char *) calloc (12,sizeof(char));
        if (tmpStr == NULL) printf("ERROR: Out of memory\n");
        if ((stop - start) > 10) printf("ERROR: Número mto grande.\n");
        memcpy (tmpStr, str + start, stop - start);

        #if _DEBUG_
        printf("SPLITv2:\t{%s-%d-%d-%d}\n", str, (atoi(tmpStr)), conta_palavras, (stop-start));
        printf("SPLITv2:\t[%s/%d]\n", tmpStr, conta_palavras);
        #endif

        vetnum[conta_palavras] = atoi(tmpStr);
        free(tmpStr);
        conta_palavras++;
        start = stop + 1;
      } else {
        printf("Error: Ultrapassou o número máximo de palavras (%d)\n",conta_palavras);
        exit (0);
      }
    }
  }

  if (tmpStr == NULL)
      printf("ERROR: Out of memory\n");
  strncpy (tmpStr, str + start, stop - start);
  return 0;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int conta_line_file ( FILE *const fin, int get_line, int nro_line, int *vet_line ) {
  char *getLine = NULL;
  int c, count, retint;
  int *nro_line_file;
  int precisao = 20;

  fseek( fin, 15, SEEK_SET );
  count = 0;
  getLine = (char *) malloc (precisao*sizeof(char));
  for( ;; ) {
    c = fgetc( fin );

    #if _DEBUG_
    printf ("CONTA_LINE_FILE:\t(%d):%c\n",count, c);
    #endif

    getLine[count] = c;
    if ( c == '\n' ) {
      getLine[count] = '\x0';

      #if _DEBUG_
      printf ("CONTA_LINE_FILE:\tLine Length = [%d] (%s)\n",count, getLine);
      #endif

      break;
    }
    count++;
    if (count > (precisao-1)) {
      printf("ERRO: Estouro de precisão!\n");
      exit(1);
    }
  }

  nro_line_file = (int *) calloc (count, sizeof(int));
  splitv2(getLine, '#', count, nro_line_file);

  #if _DEBUG_
  for (int i2=0; i2 < count; i2++)
    printf("CONTA_LINE_FILE:\tnro_line_file = %d\n", nro_line_file[i2]);
  #endif

  retint = nro_line_file[0];
  free (nro_line_file);
  free (getLine);
  return(retint);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int fileout_matriz (mymatriz *matriz, FILE *infile){
  fprintf(infile,"#Linha#Coluna#\n");
  fprintf(infile,"%d#%d#\n", matriz->lin, matriz->col);
  for (int i=0; i < matriz->lin; i++) {
    for (int j=0; j < matriz->col; j++){
      fprintf(infile,"%d,", matriz->matriz[i][j]);
    }
    fprintf(infile,"\n");
  }
  return 0;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int extrai_parametros_matriz ( FILE *const fin, int* linha,
  int *coluna, int **vet_line, int *nr_line ) {
    char *getLine = NULL;
    int firstline[2];
    int *line_matriz_file;

    *nr_line = conta_line_file (fin, 2, 1, firstline);

    #if _DEBUG_
    printf("EXTRAI_PARAMETROS_MATRIZ:\tNro de linhas = %d\n", *nr_line);
    #endif

    *vet_line = (int *) calloc ((*nr_line+2),sizeof(int));

    *nr_line = contabiliza_file (fin, *vet_line);
    getLine = get_line_file (fin, 2, *nr_line, *vet_line);
    line_matriz_file = (int *) calloc (*nr_line, sizeof(int));
    splitv2(getLine, '#', *nr_line, line_matriz_file);

    #if _DEBUG_
    printf("EXTRAI_PARAMETROS_MATRIZ:\t{{%d/%d}}\n", line_matriz_file[0], line_matriz_file[1]);
    #endif

    *linha = line_matriz_file[0];
    *coluna = line_matriz_file[1];
    free(line_matriz_file);
    free (getLine);
    return 0;
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  int contabiliza_file( FILE *const fin, int* line_length ) {
    int c, count, line = 0, vet_ind = 0;

    count = 0;
    line = 0;
    fseek( fin, 0, SEEK_SET );
    for( ;; ) {
      c = fgetc( fin );

      #if _DEBUG_
      printf ("CONTABILIZA_FILE:\t(%d):%c\n",count, c);
      #endif

      if ( c == '\n' ) {

        #if _DEBUG_
        printf ("CONTABILIZA_FILE:\tLine Length = %d\n",count);
        #endif

        line_length[vet_ind] = count;
        vet_ind++;
        count = 0;
        line++;
        // if ((vet_ind % 10) == 0) {
        // 	printf("vet_ind %d\n", vet_ind);
        // 	line_length = (int *) realloc(line_length, (vet_ind+10)*sizeof(int));
        // }
      }
      if( c == EOF )
      break;
      ++count;
    }
    return line;
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  int nr_line_file ( FILE *const fin, int *vet_line ) {
    int nr_line;

    nr_line = 1;
    vet_line = (int *) malloc ((nr_line+2)*sizeof(int));
    nr_line = conta_line_file (fin, 2, nr_line, vet_line);
    free (vet_line);
    vet_line = (int *) malloc ((nr_line+2)*sizeof(int));
    return nr_line;
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  char *get_line_file ( FILE *const fin, int get_line, int nro_line, int *vet_line ) {
    int c, count, vet_ind = 0, posicao = 0;
    char *retStr = NULL;

    #if _DEBUG_
    printf("GET_LINE_FILE:\t[%d/%d/%d]\n", nro_line, get_line, vet_line[get_line-2]);
    #endif

    count = 0;
    for (int i=0; i < (get_line-1); i++) {
      #if _DEBUG_
      printf("GET_LINE_FILE:\t(%d)=%d{%d}\n", i, vet_line[i], posicao);
      #endif
      posicao += vet_line[i];
    }

    fseek( fin, posicao+1, SEEK_SET );

    #if _DEBUG_
    printf("GET_LINE_FILE:\tposicao[%d/%d/%d/%d]\n", posicao, nro_line, get_line,vet_line[get_line-1]);
    #endif

    retStr = (char *) malloc (vet_line[get_line-1] * sizeof(char));
    for( ;; ) {
      c = fgetc( fin );

      #if _DEBUG_
      printf("GET_LINE_FILE:\t(%d):%c\n",count, c);
      #endif

      retStr[count] = c;
      if ( c == '\n' ) {
        retStr[count] = '\x0';

        #if _DEBUG_
        printf("GET_LINE_FILE:\tLine Length = %d (%s)\n",count, retStr);
        #endif

        count = -1;
        vet_ind++;

        #if _DEBUG_
        printf("GET_LINE_FILE:\t{[%s]}\n", retStr);
        #endif

        return (retStr);
      }
      if( c == EOF )
      break;
      ++count;
    }
    return NULL;
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  int filein_matriz (int **matriz, int linha, int coluna, FILE *file, int *vet_ind, int nr_line){
    char *getLine = NULL;
    int *line_matriz_file;

    #if _DEBUG_
    printf("FILEIN_MATRIZ:\tMatriz (linha/coluna) = %d/%d\n", linha, coluna);
    printf("{##########################}\n");
    #endif

    if (linha != (nr_line -2)) {
      printf("ERRO: Numero de linha da matriz nao esta correta!\n");
      exit (1);
    }

    for (int i=0; i < linha; i++) {
      getLine = get_line_file (file, i+3, nr_line, vet_ind);
      //TODO strSplit = split(getLine, ',', vet_ind[i+2]);
      line_matriz_file = (int *) calloc (vet_ind[i+2], sizeof(int));
      splitv2(getLine, ',', vet_ind[i+2], line_matriz_file);

      #if _DEBUG_
      printf("FILEIN_MATRIZ:\t{%s}\n", getLine);
      printf("FILEIN_MATRIZ:\t(%d){%d}{%d}{%d}\n", i, line_matriz_file[0], line_matriz_file[1], line_matriz_file[2]);
      #endif

      free (getLine);
      for (int j=0; j < coluna; j++){

        #if _DEBUG_
        printf("FILEIN_MATRIZ:\t(i[%d],j[%d])%d\n", i , j, line_matriz_file[j]);
        #endif

        matriz[i][j] = line_matriz_file[j];
      }
      free(line_matriz_file);
    }
    return 0;
  }
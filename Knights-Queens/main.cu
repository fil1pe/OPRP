#include <cuda.h>
#include <stdio.h>
#include "wtime.h"
#include "board.h"
#include "knights.h"

int main(int argc, char const *argv[]) {
	if (argc <= 2){
		printf("Uso: %s <número de linhas do tabuleiro> <número de cavalos>\n", argv[0]);
		return 1;
	}
	int n, m;
	n = m = atoi(argv[1]);
	int k = atoi(argv[2]);
	
	char **board;
	cudaMallocHost((void **) &board, sizeof(char*) * m);
	cudaMallocHost((void **) &(board[0]), m*n);
	for(int i=0; i<m; i++){
		board[i] = board[0] + i*n;
		for(int j=0; j<n; j++)
			board[i][j] = NO_PIECE;
	}
	chessboard b;
	b.board = board;
	b.lin = m;
	b.col = n;

	double start = wtime();

	knights(k, &b);

	//displayBoard(&b);
	displayResult(&b);
	
	printf("Tempo: %.5lf s\n", wtime() - start);

	return 0;
}

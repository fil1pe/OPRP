#include <cuda.h>
#include <stdio.h>
#include "wtime.h"
#include "board.h"
#include "knights.h"

int main(int argc, char const *argv[]) {
	// Parses arguments
	if(argc <= 2){
		printf("Uso: %s <número de linhas do tabuleiro> <número de cavalos>\n", argv[0]);
		return 1;
	}
	int n, m, k;
	n = m = atoi(argv[1]);
	k = atoi(argv[2]);

	// Allocates empty board
	chessboard board;
	board.lin = m;
	board.col = n;
	cudaMallocHost(&board.board, sizeof(char*)*m);
	cudaMallocHost(&board.board[0], m*n);
	for(int i=0; i<m; i++){
		board.board[i] = board.board[0] + i*n;
		for(int j=0; j<n; j++)
			board.board[i][j] = NO_PIECE;
	}

	// Starts stopwatch
	double start = wtime();

	// Places knights and queens
	knights(k, &board);

	// Displays result
	displayBoard(&board);
	displayResult(&board);
	printf("Tempo: %.5lf s\n", wtime() - start);

	return 0;
}

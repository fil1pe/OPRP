#include "stdio.h"
#include "board.h"

__host__ void displayBoard(chessboard *board){
	printf("\n");
	for(int i=0; i<board->lin; i++){
		for(int j=0; j<board->col; j++)
			if(board->board[i][j] != KNIGHT && board->board[i][j] != QUEEN)
				printf("\t%d;", i*(board->lin)+j+1);
			else
				printf("\t%c;", board->board[i][j]);
		printf("\n");
	}
	printf("\n");
}

__host__ void displayResult(chessboard *board){
	for(int i=0; i<board->lin; i++)
		for(int j=0; j<board->col; j++)
			if(board->board[i][j] == KNIGHT)
				printf("K%d;", i*(board->lin)+j+1);
			else if(board->board[i][j] == QUEEN)
				printf("Q%d;", i*(board->lin)+j+1);
	printf("\n");
}
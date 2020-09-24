#include <cuda.h>
#include "board.h"

// Places queen at (i, j)
__global__ void place_queen(char *board, int i, int j, int m, int n){
	int k = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// Places queen
	if (k == 1) board[i*n + j] = QUEEN;

	// Marks the attacking positions of the queen
	if (i+k < m)
		board[(i+k)*n + j] = QUEEN_ATTACK;
	if (i-k >= 0)
		board[(i-k)*n + j] = QUEEN_ATTACK;
	if (j+k < n)
		board[i*n + j+k] = QUEEN_ATTACK;
	if (j-k >= 0)
		board[i*n + j-k] = QUEEN_ATTACK;
	if (i+k < m && j+k < n)
		board[(i+k)*n + j+k] = QUEEN_ATTACK;
	if (i+k < m && j-k >= 0)
		board[(i+k)*n + j-k] = QUEEN_ATTACK;
	if (i-k >= 0 && j+k < n)
		board[(i-k)*n + j+k] = QUEEN_ATTACK;
	if (i-k >= 0 && j-k >= 0)
		board[(i-k)*n + j-k] = QUEEN_ATTACK;
}

// Places queens
__host__ void queens(int qui, int quj, chessboard *board, char *board_dev, char **skip) {
	// Allocates auxiliary matrix that tells whether to skip cells
	if(skip == NULL){
		cudaMallocHost(&skip, sizeof(char*)*board->lin);
		cudaMallocHost(&skip[0], board->lin * board->col);
		for(int i=0; i<board->lin; i++){
			skip[i] = skip[0] + i*board->col;
			for(int j=0; j<board->col; j++)
				skip[i][j] = 0;
		}
	}

	for(int i=qui; i<board->lin; i++){
		for(int j=quj; j<board->col; j++)
			// If (i, j) is free to place queen, does it
			if(board->board[i][j] == NO_PIECE){
				dim3 dimBlock (1);dim3 dimThreads(board->lin); // MELHORAR
				place_queen<<<dimBlock, dimThreads>>>(board_dev, i, j, board->lin, board->col);
				cudaDeviceSynchronize();
				cudaMemcpy(board->board[0], board_dev, board->lin * board->col, cudaMemcpyDeviceToHost);

			// If (i, j) is marked in order to not place queen, calls recursion 
			}else if((board->board[i][j] == QUEEN_ATTACK || board->board[i][j] == DONT_PLACE_QUEEN) && !skip[i][j]){
				skip[i][j] = 1;
				queens(i, j, board, board_dev, skip);
			}
	}
}
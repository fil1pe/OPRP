#include <cuda.h>
#include "board.h"

// Marks positions where no other queen must be placed
// in order to ensure this queen is safe
__global__ void attack_queens(char *board, int i, int j, int m, int n){
	int k = blockIdx.x * blockDim.x + threadIdx.x + 1;

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
__host__ int queens(int qui, int quj, chessboard *board) {
	for(int i=qui; i<board->lin; i++){
		for(int j=quj; j<board->col; j++)
			// If (i, j) is free to place queen,
			// do it
			if(board->board[i][j] == NO_PIECE){
				char *board_dev;
				cudaMalloc(&board_dev, board->lin * board->col);
				cudaMemcpy(board_dev, board->board[0], board->lin * board->col, cudaMemcpyHostToDevice);
				dim3 dimBlock (1);dim3 dimThreads(board->lin); // MELHORAR
				attack_queens <<<dimBlock, dimThreads>>>(board_dev, i, j, board->lin, board->col);
				cudaDeviceSynchronize();
				cudaMemcpy(board->board[0], board_dev, board->lin * board->col, cudaMemcpyDeviceToHost);
				board->board[i][j] = QUEEN;

			// If (i, j) is marked in order to not place queen,
			// call recursion 
			}else if(board->board[i][j] == QUEEN_ATTACK || board->board[i][j] == DONT_PLACE_QUEEN){
				board->board[i][j] = SKIP;
				queens(i, j, board);
			}
	}

	return 0;
}
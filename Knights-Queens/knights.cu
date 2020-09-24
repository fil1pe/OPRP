#include <cuda.h>
#include "board.h"
#include "queens.h"

// Places knight at (i, j)
__global__ void place_knight(char *board, int i, int j, int m, int n){
	int k = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// Only one thread needs to run this block
	if(k == 1){
		// Places knight
		board[i*n + j] = KNIGHT;

		// Marks the attacking positions of the knight
		if (i+2 < m && j-1 >= 0)
			board[(i+2)*n + j-1] = KNIGHT_ATTACK;
		if (i-2 >= 0 && j-1 >= 0)
			board[(i-2)*n + j-1] = KNIGHT_ATTACK;
		if (i+2 < m && j+1 < n)
			board[(i+2)*n + j+1] = KNIGHT_ATTACK;
		if (i-2 >= 0 && j+1 < n)
			board[(i-2)*n + j+1] = KNIGHT_ATTACK;
		if (i+1 < m && j+2 >= 0)
			board[(i+1)*n + j+2] = KNIGHT_ATTACK;
		if (i-1 >= 0 && j+2 >= 0)
			board[(i-1)*n + j+2] = KNIGHT_ATTACK;
		if (i+1 < m && j-2 >= 0)
			board[(i+1)*n + j-2] = KNIGHT_ATTACK;
		if (i-1 >= 0 && j-2 >= 0)
			board[(i-1)*n + j-2] = KNIGHT_ATTACK;
	}

	// Marks positions where no queen must be placed
	// in order to ensure the knight is safe
	if (i+k < m && board[(i+k)*n + j] == NO_PIECE)
		board[(i+k)*n + j] = DONT_PLACE_QUEEN;
	if (i-k >= 0 && board[(i-k)*n + j] == NO_PIECE)
		board[(i-k)*n + j] = DONT_PLACE_QUEEN;
	if (j+k < n && board[i*n + j+k] == NO_PIECE)
		board[i*n + j+k] = DONT_PLACE_QUEEN;
	if (j-k >= 0 && board[i*n + j-k] == NO_PIECE)
		board[i*n + j-k] = DONT_PLACE_QUEEN;
	if (i+k < m && j+k < n && board[(i+k)*n + j+k] == NO_PIECE)
		board[(i+k)*n + j+k] = DONT_PLACE_QUEEN;
	if (i+k < m && j-k >= 0 && board[(i+k)*n + j-k] == NO_PIECE)
		board[(i+k)*n + j-k] = DONT_PLACE_QUEEN;
	if (i-k >= 0 && j+k < n && board[(i-k)*n + j+k] == NO_PIECE)
		board[(i-k)*n + j+k] = DONT_PLACE_QUEEN;
	if (i-k >= 0 && j-k >= 0 && board[(i-k)*n + j-k] == NO_PIECE)
		board[(i-k)*n + j-k] = DONT_PLACE_QUEEN;
}

// Places k knights
__host__ void knights(int k, chessboard *board, char *board_dev, dim3 dimBlock, dim3 dimThreads){
	// Allocates board on the device
	if(board_dev == NULL){
		cudaMalloc(&board_dev, board->lin * board->col);
		cudaMemcpy(board_dev, board->board[0], board->lin * board->col, cudaMemcpyHostToDevice);
	}

	// If no knight is to be placed, places queens
	if(k == 0){
		queens(0, 0, board, board_dev, NULL, dimBlock, dimThreads);
		return;
	}

	// Counter for the number of knights already placed
	int count = 0;

	// Places knights in linear ordering
	for(int i=0; i<board->lin; i++)
		for(int j=0; j<board->col; j++)
			// If (i, j) is free to place knight, does it
			if(board->board[i][j] == NO_PIECE || board->board[i][j] == DONT_PLACE_QUEEN){
				place_knight<<<dimBlock, dimThreads>>>(board_dev, i, j, board->lin, board->col);
				cudaDeviceSynchronize();
				cudaMemcpy(board->board[0], board_dev, board->lin * board->col, cudaMemcpyDeviceToHost);
				
				// If already placed k knights, it's time to place queens
				if(++count == k){
					knights(0, board, board_dev, dimBlock, dimThreads);
					return;
				}
			}
}
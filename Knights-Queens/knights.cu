#include <cuda.h>
#include "board.h"
#include "queens.h"

// Marks positions where no queen must be placed
// in order to ensure the knight is safe
__global__ void attack_knights(char *board, int i, int j, int m, int n){
	int k = blockIdx.x * blockDim.x + threadIdx.x + 1;

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

// Marks the attacking positions relative to the knight at (i, j)
__host__ void attack_knights_host(char *board, int i, int j, int m, int n){
	if (i+2 < m && j - 1 >= 0)
		board[(i+2)*n + j-1] = KNIGHT_ATTACK;
	if (i-2 >= 0 && j - 1 >= 0)
		board[(i-2)*n + j-1] = KNIGHT_ATTACK;
	if (i+2 < m && j+1 < n)
		board[(i+2)*n + j+1] = KNIGHT_ATTACK;
	if (i-2 >= 0 && j + 1 < n)
		board[(i-2)*n + j+1] = KNIGHT_ATTACK;
	if (i+1 < m && j +2 >= 0)
		board[(i+1)*n + j+2] = KNIGHT_ATTACK;
	if (i-1 >= 0 && j + 2 >= 0)
		board[(i-1)*n + j+2] = KNIGHT_ATTACK;
	if (i+1 < m && j - 2 >= 0)
		board[(i+1)*n + j-2] = KNIGHT_ATTACK;
	if (i-1 >= 0 && j - 2 >= 0)
		board[(i-1)*n + j-2] = KNIGHT_ATTACK;

	// Tells GPU to block queens
	char *board_dev;
	cudaMalloc(&board_dev, m*n);
	cudaMemcpy(board_dev, board, m*n, cudaMemcpyHostToDevice);
	dim3 dimBlock (1);dim3 dimThreads(m); // MELHORAR
	attack_knights <<<dimBlock, dimThreads>>>(board_dev, i, j, m, n);
	cudaDeviceSynchronize();
	cudaMemcpy(board, board_dev, m*n, cudaMemcpyDeviceToHost);
}

// Places k knights
__host__ void knights(int k, chessboard *board){
	// If no knight is to be placed,
	// places queens
	if(k == 0){
		queens(0, 0, board);
		return;
	}

	// Counter for the number of knights already placed
	int cont = 0;

	// Places knights in linear ordering
	for(int i=0; i<board->lin; i++)
		for(int j=0; j<board->col; j++)
			if(board->board[i][j] == NO_PIECE || board->board[i][j] == DONT_PLACE_QUEEN){
				attack_knights_host(board->board[0], i, j, board->lin, board->col);
                board->board[i][j] = KNIGHT;
				
				// If already placed k knights,
				// it's time to place queens
				if(++cont == k){
					knights(0, board);
					return;
				}
			}
}
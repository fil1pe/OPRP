#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define NO_PIECE			'_'
#define KNIGHT				'K'
#define QUEEN				'Q'
#define KNIGHT_ATTACK		'k'
#define QUEEN_ATTACK		'q'
#define DONT_PLACE_QUEEN	'*'
#define SKIP				'j'

typedef struct {
	char** board;
	int lin, col;
} chessboard;

__host__ double wtime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + (double) t.tv_usec / 1000000;
}

// Display board
__host__ void displayBoard(chessboard* board){
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

// Display positions of queens and knights
__host__ void displayResult(chessboard* board){
	for(int i=0; i<board->lin; i++)
		for(int j=0; j<board->col; j++)
			if(board->board[i][j] == KNIGHT)
				printf("K%d;", i*(board->lin)+j+1);
			else if(board->board[i][j] == QUEEN)
				printf("Q%d;", i*(board->lin)+j+1);
	printf("\n");
}

__global__ void attack_queens(char* board, int i, int j, int m, int n){
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

/* Place queens on board such that they
don't attack each other or the knights */
__host__ int queens(int qui, int quj, chessboard* board) {
	for(int i=qui; i<board->lin; i++){
		for(int j=quj; j<board->col; j++)
			/* Is it possible to place a queen at (i, j)? */
			if(board->board[i][j] == NO_PIECE){
				/* Place a queen at (i, j) */
				char* board_dev;
				cudaMalloc((void**) &board_dev, board->lin*board->col);
				cudaMemcpy(board_dev, board->board[0], board->lin*board->col, cudaMemcpyHostToDevice);
				dim3 dimBlock (1);dim3 dimThreads(board->lin); // MELHORAR
				attack_queens<<< dimBlock, dimThreads>>>(board_dev, i, j, board->lin, board->col);
				cudaDeviceSynchronize();
				cudaMemcpy (board->board[0], board_dev, board->lin*board->col, cudaMemcpyDeviceToHost);
				board->board[i][j] = QUEEN;
			}else if(board->board[i][j] == QUEEN_ATTACK || board->board[i][j] == DONT_PLACE_QUEEN){
				board->board[i][j] = SKIP;
				queens(i, j, board);
			}
	}

	return 0;
}

__global__ void attack_knights(char* board, int i, int j, int m, int n){
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

__host__ void attack_knights_host(char* board, int i, int j, int m, int n){
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
	
	char* board_dev;
	cudaMalloc((void**) &board_dev, m*n);
	cudaMemcpy(board_dev, board, m*n, cudaMemcpyHostToDevice);
	dim3 dimBlock (1);dim3 dimThreads(m); // MELHORAR
	attack_knights<<< dimBlock, dimThreads>>>(board_dev, i, j, m, n);
	cudaDeviceSynchronize();
	cudaMemcpy (board, board_dev, m*n, cudaMemcpyDeviceToHost);
}

/* Place k knights on board such that
they don't attack each other */
__host__ void knights(int k, chessboard* board){
	/* Aren't there knights left to place? */
	if(k == 0){
		queens(0, 0, board);
		return;
	}

	int cont = 0;

	for(int i=0; i<board->lin; i++)
		for(int j=0; j<board->col; j++)
			/* Is it possible to place a knight at (i, j)? */
			if(board->board[i][j] == NO_PIECE || board->board[i][j] == DONT_PLACE_QUEEN){
				attack_knights_host(board->board[0], i, j, board->lin, board->col);
				board->board[i][j] = KNIGHT;

				/* Already placed k knights? */
				if(++cont == k)
					return knights(0, board);
			}
}

// Driver code
int main(int argc, char const *argv[]) {
	// Parse arguments
	if (argc <= 2){
		printf("Uso: %s <número de linhas do tabuleiro> <número de cavalos>\n", argv[0]);
		return 1;
	}
	int n, m;
	n = m = atoi(argv[1]);
	int k = atoi(argv[2]);

	// Generate empty board
	char** board;
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

	// Initialize stopwatch
	double start = wtime();

	// Place knights and queens
	knights(k, &b);

	// Print results
	//displayBoard(&b);
	displayResult(&b);

	// Show runtime
	printf("Tempo: %.5lf s\n", wtime() - start);

	return 0;
}

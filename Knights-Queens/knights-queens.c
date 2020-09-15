#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>

#include <omp.h>

#define NO_PIECE			'_'
#define KNIGHT				'K'
#define QUEEN				'Q'
#define KNIGHT_ATTACK		'k'
#define QUEEN_ATTACK		'q'
#define DONT_PLACE_QUEEN	'*'
#define SKIP				'j'

double wtime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + (double) t.tv_usec / 1000000;
}

// min{a, b}
int min(int a, int b){
	if (a > b) return b;
	return a;
}

/* m x n is the board dimension */
int m, n, nth;

// Display board
void displayBoard(char** board){
	printf("\n");
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++)
			if(board[i][j] != KNIGHT && board[i][j] != QUEEN)
				printf("\t%d;", i*m+j+1);
			else
				printf("\t%c;", board[i][j]);
		printf("\n");
	}
	printf("\n");
}

// Display positions of queens and knights
void displayResult(char** board){
	for(int i=0; i<m; i++)
		for(int j=0; j<n; j++)
			if(board[i][j] == KNIGHT)
				printf("K%d;", i*m+j+1);
			else if(board[i][j] == QUEEN)
				printf("Q%d;", i*m+j+1);
	printf("\n");
}

/* Mark attacking positions of queen placed
at board[oi][oj] */
void attack_queens(int oi, int oj, char** board) {
	int i, j, k, x1, x2, y1, y2, max1, max2, aux;

	aux = min(oi, oj);
	x1 = oi - aux;
	y1 = oj - aux;

	max1 = m-x1;
	if(y1 + max1 > n)
		max1 = n-y1;

	aux = min(oi, n - 1 - oj);
	x2 = oi - aux;
	y2 = oj + aux;

	max2 = m-x2;
	if(y2 - max2 < 0)
		max2 = y2;

	#pragma omp parallel num_threads(nth)
	{

	/* Mark the attacking positions in the vertical */
	#pragma omp for private(i) nowait
	for(i=0; i<m; i++)
		board[i][oj] = QUEEN_ATTACK;

	/* Mark the attacking positions in the horizontal */
	#pragma omp for private(j) nowait
	for(j=0; j<n; j++)
		board[oi][j] = QUEEN_ATTACK;

	/* Mark the attacking positions in the main diagonal */
	#pragma omp for private(k) nowait
	for(k=0; k<max1; k++)
		board[x1+k][y1+k] = QUEEN_ATTACK;

	/* Mark the attacking positions in the anti-diagonal */
	#pragma omp for private(k) nowait
	for(k=0; k<max2; k++)
		board[x2+k][y2-k] = QUEEN_ATTACK;

	}
}

int knight_movs[]={2,1,2,-1,-2,1,-2,-1,1,2,1,-2,-1,2,-1,-2};

/* Mark all the attacking and protected positions of
knight placed at board[i][j] */
void attack_knights(int oi, int oj, char** board) {
	int i, j, k, x1, x2, y1, y2, max1, max2, aux;

	aux = min(oi, oj);
	x1 = oi - aux;
	y1 = oj - aux;

	max1 = m-x1;
	if(y1 + max1 > n)
		max1 = n-y1;

	aux = min(oi, n - 1 - oj);
	x2 = oi - aux;
	y2 = oj + aux;

	max2 = m-x2;
	if(y2 - max2 < 0)
		max2 = y2;

	#pragma omp parallel num_threads(nth)
	{

	#pragma omp for private(i) nowait
	for(i=0; i<16; i+=2){
		int x = oi + knight_movs[i],
		y = oj - knight_movs[i+1];
		if(x < m && x >= 0 && y < n && y >= 0)
			board[x][y] = KNIGHT_ATTACK;
	}

	/* Mark the attacking positions in the vertical */
	#pragma omp for private(i) nowait
	for(i=0; i<m; i++)
		if(board[i][oj] == NO_PIECE)
			board[i][oj] = DONT_PLACE_QUEEN;

	/* Mark the attacking positions in the horizontal */
	#pragma omp for private(j) nowait
	for(j=0; j<n; j++)
		if(board[oi][j] == NO_PIECE)
			board[oi][j] = DONT_PLACE_QUEEN;

	/* Mark the attacking positions in the main diagonal */
	#pragma omp for private(k) nowait
	for(k=0; k<max1; k++)
		if(board[x1+k][y1+k] == NO_PIECE)
			board[x1+k][y1+k] = DONT_PLACE_QUEEN;

	/* Mark the attacking positions in the anti-diagonal */
	#pragma omp for private(k) nowait
	for(k=0; k<max2; k++)
		if(board[x2+k][y2+k] == NO_PIECE)
			board[x2+k][y2-k] = DONT_PLACE_QUEEN;

	}
}

// Place knight or queen at board[i][j]
void place(int i, int j, char k, char** board){
	/* Mark all the attacking positions of
	the new piece on the new board */
	if (k == KNIGHT)
		attack_knights(i, j, board);
	else
		attack_queens(i, j, board);

	/* Place the knight/queen */
	board[i][j] = k;
}

/* Place queens on board such that they
don't attack each other or the knights */
int queens(int qui, int quj, char*** board) {
	for(int i=qui; i<m; i++){
		for(int j=quj; j<n; j++)
			/* Is it possible to place a queen at (i, j)? */
			if((*board)[i][j] == NO_PIECE){
				/* Place a queen at (i, j) */
				place(i, j, QUEEN, *board);
			}else if((*board)[i][j] == QUEEN_ATTACK || (*board)[i][j] == DONT_PLACE_QUEEN){
				(*board)[i][j] = SKIP;
				queens(i, j, board);
			}
	}

	return 0;
}

/* Place k knights on board such that
they don't attack each other */
void knights(int k, char** board){
	/* Aren't there knights left to place? */
	if(k == 0){
		queens(0, 0, &board);
		displayBoard(board);
		displayResult(board);
		return;
	}

	int cont = 0;

	for(int i=0; i<m; i++)
		for(int j=0; j<n; j++)
			/* Is it possible to place a knight at (i, j)? */
			if(board[i][j] == NO_PIECE || board[i][j] == DONT_PLACE_QUEEN){
				place(i, j, KNIGHT, board);

				/* Already placed k knights? */
				if(++cont == k)
					return knights(0, board);
			}
}

// Driver code
int main (int argc, char *argv[]){
	// Parse arguments
	if (argc <= 2){
		printf("Uso: %s <número de linhas do tabuleiro> <número de cavalos>\n", argv[0]);
		return 1;
	}
	n = m = atoi(argv[1]);
	int k = atoi(argv[2]);
	nth = 1;

	// Generate empty board
	char** board = (char**) malloc(sizeof(char*) * m);
	board[0] = (char*) malloc(m*n);
	for(int i=1; i<m; i++)
		board[i] = board[0] + i * n;
	memset(board[0], NO_PIECE, m*n);

	// Initialize stopwatch
	double start = wtime();

	// Place knights and queens
	knights(k, board);

	// Show runtime
	printf("Tempo: %.5lf s\n", wtime() - start);

	return 0;
}

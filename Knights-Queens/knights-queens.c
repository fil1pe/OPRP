#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NO_PIECE		'_'
#define KNIGHT			'K'
#define QUEEN			'Q'
#define KNIGHT_ATTACK	'A'
#define QUEEN_ATTACK	'a'
#define DONT_PLACE		'*'

/*	m x n is the board dimension
	k is the number of knights to be placed on board
	count is the number of possible solutions */
int m, n, k;
int count = 0;

// Display board
void displayBoard(char** board) {
	printf("\n");
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			if (board[i][j] != KNIGHT && board[i][j] != QUEEN)
				printf("\t%d;", i*m+j+1);
			else
				printf("\t%c;", board[i][j]);
		printf("\n");
	}
	printf("\n");
}

// Display positions of queens and knights
void displayResult(char** board) {
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			if (board[i][j] == KNIGHT)
				printf("K%d;", i*m+j+1);
			else if (board[i][j] == QUEEN)
				printf("Q%d;", i*m+j+1);
	printf("\n");
}

/* Mark all the attacking positions of
the knight placed at board[i][j] */
int attack_knights(int i, int j, char** board) {
	if (i + 2 < m && j - 1 >= 0)
		board[i + 2][j - 1] = KNIGHT_ATTACK;
	if (i - 2 >= 0 && j - 1 >= 0)
		board[i - 2][j - 1] = KNIGHT_ATTACK;
	if (i + 2 < m && j + 1 < n)
		board[i + 2][j + 1] = KNIGHT_ATTACK;
	if (i - 2 >= 0 && j + 1 < n)
		board[i - 2][j + 1] = KNIGHT_ATTACK;
	if (i + 1 < m && j + 2 < n)
		board[i + 1][j + 2] = KNIGHT_ATTACK;
	if (i - 1 >= 0 && j + 2 < n)
		board[i - 1][j + 2] = KNIGHT_ATTACK;
	if (i + 1 < m && j - 2 >= 0)
		board[i + 1][j - 2] = KNIGHT_ATTACK;
	if (i - 1 >= 0 && j - 2 >= 0)
		board[i - 1][j - 2] = KNIGHT_ATTACK;
	
	return 0;
}

/* Mark attacking positions of the queen placed
at board[oi][oj] */
int attack_queens(int oi, int oj, char** board) {
	int i, j;

	/* Mark the attacking positions in the vertical */
	i = oi;
	j = oj;
	while (i > 0) {
		if (board[i-1][j] == QUEEN || board[i-1][j] == KNIGHT)
			return 1;
		board[i-1][j] = QUEEN_ATTACK;
		i--;
	}

	i = oi;
	j = oj;
	while (i+1 < m) {
		if (board[i+1][j] == QUEEN || board[i+1][j] == KNIGHT)
			return 1;
		board[i+1][j] = QUEEN_ATTACK;
		i++;
	}

	/* Mark the attacking positions in the horizontal */
	i = oi;
	j = oj;
	while (j > 0) {
		if (board[i][j-1] == QUEEN || board[i][j-1] == KNIGHT)
			return 1;
		board[i][j-1] = QUEEN_ATTACK;
		j--;
	}

	i = oi;
	j = oj;
	while (j+1 < n) {
		if (board[i][j+1] == QUEEN || board[i][j+1] == KNIGHT)
			return 1;
		board[i][j+1] = QUEEN_ATTACK;
		j++;
	}

	/* Mark the attacking positions in the diagonal 1 */
	i = oi;
	j = oj;
	while (i > 0 && j > 0) {
		if (board[i-1][j-1] == QUEEN || board[i-1][j-1] == KNIGHT)
			return 1;
		board[i-1][j-1] = QUEEN_ATTACK;
		i--;
		j--;
	}

	i = oi;
	j = oj;
	while (i+1 < m && j+1 < n) {
		if (board[i+1][j+1] == QUEEN || board[i+1][j+1] == KNIGHT)
			return 1;
		board[i+1][j+1] = QUEEN_ATTACK;
		i++;
		j++;
	}

	/* Mark the attacking positions in the diagonal 2 */
	i = oi;
	j = oj;
	while (i > 0 && j < n) {
		if (board[i-1][j+1] == QUEEN || board[i-1][j+1] == KNIGHT)
			return 1;
		board[i-1][j+1] = QUEEN_ATTACK;
		i--;
		j++;
	}

	i = oi;
	j = oj;
	while (i+1 < m && j > 0) {
		if (board[i+1][j-1] == QUEEN || board[i+1][j-1] == KNIGHT)
			return 1;
		board[i+1][j-1] = QUEEN_ATTACK;
		i++;
		j--;
	}

	return 0;
}

// Place knight or queen at board[i][j]
int place(int i, int j, char k, char** board, char** new_board){
	// Copy board
	new_board[0] = (char*) malloc(sizeof(char) * m*n);
	for (int x=0; x<m; x++) {
		new_board[x] = new_board[0] + x * n;
		for (int y=0; y<n; y++)
			new_board[x][y] = board[x][y];
	}

	// Place piece on the new board
	new_board[i][j] = k;

	/* Mark all the attacking positions of
	the newly placed piece on the new board */
	return k == KNIGHT ?
		attack_knights(i, j, new_board) :
		attack_queens(i, j, new_board);
}

/* Place queens on board such that they
don't attack each other or the knights */
void queens(int qui, int quj, char** board) {
	if (qui*m+quj+1 >= m*n)
		return;
	for (int i=qui; i<m; i++)
		for (int j=quj; j<n; j++)
			/* Is it possible to place a queen at board[i][j]? */
			if (board[i][j] == NO_PIECE) {
				/* Create a new board and place a queen on it */
				char** new_board = (char**) malloc(sizeof(char*) * m);
				if (place(i, j, QUEEN, board, new_board)) {
					board[i][j] = DONT_PLACE;
					queens(i, j, board);
				} else {
					board[i][j] = QUEEN;
					queens(i, j, new_board);
				}

				// Free memory
				free(new_board[0]);
				free(new_board);
			}
}

/* Place k knights on board such that
they don't attack each other */
void knights(int k, int sti, int stj, char** board) {
	/* If there are no knights left to be placed,
	display the board and increment the count */
	if (k == 0) {
		queens(0, 0, board);
		displayBoard(board);
		displayResult(board);
		count++;
		return;
	}

	for (int i=sti; i<m; i++) {
		for (int j=stj; j<n; j++) {
			/* Is it possible to place a knight at board[i][j]? */
			if (board[i][j] == NO_PIECE) {
				/* Create a new board and place a knight on it */
				char** new_board = (char**) malloc(sizeof(char*) * m);
				place(i, j, KNIGHT, board, new_board);

				/* Call the function recursively for (k-1) leftover knights */
				knights(k - 1, i, j, new_board);

				// Free memory
				free(new_board[0]);
				free(new_board);
			}
		}
		stj = 0;
	}
}

// Driver code
int main(int argc, char *argv[]){
	// Parse arguments
	if (argc <= 2) {
		printf("Uso: %s <número de linhas do tabuleiro> <número de cavalos>\n", argv[0]);
		return 1;
	}
	n = m = atoi(argv[1]);

	// Generate empty board
	char** board = (char**) malloc(sizeof(char*) * m);
	board[0] = (char*) malloc(sizeof(char) * m*n);
	memset(board[0], NO_PIECE, m*n);
	for (int i=1; i<m; i++)
		board[i] = board[0] + i * n;

	// Place knights and queens
	knights(atoi(argv[2]), 0, 0, board);

	printf("\nNúmero de soluções: %d\n", count);
	return 0;
}

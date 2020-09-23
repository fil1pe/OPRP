// C++ implementation of the above approach
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
using namespace std;

//http://mathworld.wolfram.com/KnightsProblem.html

/* m*n is the board dimension
k is the number of knights to be placed on board
count is the number of possible solutions */
int m, n, k;
int count = 0;

double start;

double wtime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + (double) t.tv_usec / 1000000;
}

/* This function is used to create an empty m*n board */
void makeBoard(char** board)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			board[i][j] = '_';
		}
	}
}

/* This function displays our board */
void displayBoard(char** board)
{
	cout << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << "\t" << board[i][j] << " ";
		}
		cout << endl;
	}
}

/* This function displays our board */
void displayBoardCoordenatesMatrix(char** board)
{
	cout << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if ((board[i][j] != 'K') && (board[i][j] != 'Q')) {
				cout << "\t" << (i*m)+(j+1) << ";" ;
			} else {
				cout << "\t" << board[i][j] << ";" ;
			}
		}
		cout << endl;
	}
	cout << endl;
}

/* This function displays our board */
void displayResult(char** board)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (board[i][j] == 'K') {
				cout << "K" << (i*m)+(j+1) << ";" ;
			}
			if (board[i][j] == 'Q') {
				cout << "Q" << (i*m)+(j+1) << ";" ;
			}
		}
	}
	cout << endl;
}


/* This function displays our board */
void displayBoardCoordenates(char** board)
{
	cout << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if ((board[i][j] == 'K') || (board[i][j] == 'Q')) {
				//cout << m << board[i][j] << i << j << (i*m)+(j+1) << ";" ;
				cout << (i*m)+(j+1) << ";" ;
			}
		}
	}
	cout << endl;
}



/* This function marks all the attacking
position of a knight placed at board[i][j]
position */
void attack(int i, int j, char a,
	char** board)
	{

		/* conditions to ensure that the
		block to be checked is inside the boardvoid displayBoardCoordenatesMatrix(char** board, char a) */
		if ((i + 2) < m && (j - 1) >= 0) {
			board[i + 2][j - 1] = a;
		}
		if ((i - 2) >= 0 && (j - 1) >= 0) {
			board[i - 2][j - 1] = a;
		}
		if ((i + 2) < m && (j + 1) < n) {
			board[i + 2][j + 1] = a;
		}
		if ((i - 2) >= 0 && (j + 1) < n) {
			board[i - 2][j + 1] = a;
		}
		if ((i + 1) < m && (j + 2) < n) {
			board[i + 1][j + 2] = a;
		}
		if ((i - 1) >= 0 && (j + 2) < n) {
			board[i - 1][j + 2] = a;
		}
		if ((i + 1) < m && (j - 2) >= 0) {
			board[i + 1][j - 2] = a;
		}
		if ((i - 1) >= 0 && (j - 2) >= 0) {
			board[i - 1][j - 2] = a;
		}
	}

	int attackqueens(int oi, int oj, char a,
		char** board)
		{
			int i,j;

			i = oi;
			j = oj;
			while ((i > 0) && (j > 0)) {
				if ((board[i-1][j-1] != 'K') || (board[i-1][j-1] != 'A') || (board[i-1][j-1] != 'Q') || (board[i-1][j-1] != 'q')){
					//board[i-1][j-1] = a;
					if ((board[i-1][j-1] == 'K')) {
						return(1);
					} else {
						board[i-1][j-1] = a;
					}
				}
				i--;
				j--;
			}

			i = oi;
			j = oj;
			while (i > 0) {
				if (board[i-1][j] == 'Q') {
					return(1);
				}
				if (board[i-1][j] == 'K') {
					return(1);
				} else {
					board[i-1][j] = a;
				}
				i--;
			}

			i = oi;
			j = oj;
			while ((i > 0) && (j < n)) {
				if (board[i-1][j+1] == 'Q') {
					return(1);
				}
				if (board[i-1][j+1] == 'K') {
					return(1);
				} else {
					board[i-1][j+1] = a;
				}
				i--;
				j++;
			}

			i = oi;
			j = oj;
			while (j > 0) {
				if (board[i][j-1] == 'Q') {
					return(1);
				}
				if (board[i][j-1] == 'K') {
					return(1);
				} else {
						board[i][j-1] = a;
				}
				j--;
			}

			i = oi;
			j = oj;
			while (j+1 < n) {
				if (board[i][j+1] == 'Q') {
					return(1);
				}
				if (board[i][j+1] == 'K') {
					return(1);
				} else {
				   board[i][j+1] = a;
			  }
				j++;
			}

			i = oi;
			j = oj;
			while ((i+1 < m) && (j > 0)) {
				if (board[i+1][j-1] == 'Q') {
					return(1);
				}
				if (board[i+1][j-1] == 'K') {
					return(1);
				} else {
				   board[i+1][j-1] = a;
			  }
				i++;
				j--;
			}

			i = oi;
			j = oj;
			while (i+1 < m) {
				if (board[i+1][j] == 'Q') {
					return(1);
				}
				if (board[i+1][j] == 'K') {
					return(1);
				} else {
				   board[i+1][j] = a;
			  }
				i++;
			}

			i = oi;
			j = oj;
			while ((i+1 < m) && (j+1 < n)) {
				if (board[i+1][j+1] == 'Q') {
					return(1);
				}
				if (board[i+1][j+1] == 'K') {
					return(1);
				} else {
				   board[i+1][j+1] = a;
			  }
				i++;
				j++;
			}
			return(0);
		}


		/* If the position is empty,
		place the knight */
		bool canPlace(int i, int j, char** board)
		{
			if (board[i][j] == '_')
			return true;
			else
			return false;
		}

		/* Place the knight at [i][j] position
		on board */
		int place(int i, int j, char k, char a,
			char** board, char** new_board)
			{
				/* Copy the configurations of
				old board to new board */
				for (int y = 0; y < m; y++) {
					for (int z = 0; z < n; z++) {
						new_board[y][z] = board[y][z];
						//printf("%d%d %c\n", y, z, new_board[y][z]);
					}
				}

				/* Place the knight at [i][j]
				position on new board */
				new_board[i][j] = k;
				//displayBoard(new_board);
				//printf("%c(%d%d)", new_board[i][j], i, j);

				/* Mark all the attacking positions
				of newly placed knight on the new board */
				if (k == 'K') {
					attack(i, j, a, new_board);
				} else {
					return(attackqueens(i, j, a, new_board));
				}
				return (0);
			}

			void queens (int qui, int quj, char ** board) {
				if ((qui*m)+(quj+1) >= (m*n)) {
					//displayBoard(board);
					//displayBoardCoordenates(board,'Q');
				} else {
					//printf("%d-%d\n", qui,quj);
					for (int i = qui; i < m; i++) {
						for (int j = quj; j < n; j++) {
							//printf("%d--%d\n", i,j);
							if (canPlace(i,j,board)) {

								//cout << "canPlace" << i << j << endl;

								/* Create a a new board and place the
								new queen on it */
								char** new_board = new char*[m];
								for (int x = 0; x < m; x++) {
									new_board[x] = new char[n];
								}
								if (place(i, j, 'Q', 'a', board, new_board)) {
									/* Call the function recursively for
									(Q-1) leftover queens */
									//printf("Try to place %d:%d\n", i,j);
									board[i][j] = '*';
									queens(i, j, board);
								} else {
									/* Call the function recursively for
									(Q-1) leftover queens */
									//printf("Place queen on %d:%d\n", i,j);
									board[i][j] = 'Q';
									queens(i, j, new_board);
								}
								//displayBoard(board);

								/* Delete the new board
								to free up the memory */
								for (int x = 0; x < m; x++) {
									delete[] new_board[x];
								}
								delete[] new_board;

							}
						}
					}
				}
			}

			/* Function for placing knights on board
			such that they don't attack each other */
			void kkn(int k, int sti, int stj, char** board)
			{

				/* If there are no knights left to be placed,
				display the board and increment the count */
				if (k == 0) {
					//displayBoard(board);
					queens(0,0,board);
					//displayBoard(board);
					//displayBoardCoordenates(board,'x');
					displayBoardCoordenatesMatrix(board);
					displayResult(board);
					printf("Tempo: %.5lf s\n", wtime() - start);
					//displayBoardCoordenates(board);
					exit (1);
					count++;
				}
				else {

					/* Loop for checking all the	positions on m*n board */
					for (int i = sti; i < m; i++) {
						for (int j = stj; j < n; j++) {

							/* Is it possible to place knight at [i][j] position on board? */
							if (canPlace(i, j, board)) {

								/* Create a a new board and place the	new knight on it */
								char** new_board = new char*[m];
								for (int x = 0; x < m; x++) {
									new_board[x] = new char[n];
								}
								place(i, j, 'K', 'A', board, new_board);
								//displayBoard(board);

								/* Call the function recursively for (k-1) leftover knights */
								kkn(k - 1, i, j, new_board);

								/* Delete the new board	to free up the memory */
								for (int x = 0; x < m; x++) {
									delete[] new_board[x];
								}
								delete[] new_board;
							}
						}
						stj = 0;
					}
				}
			}

			// Driver code
			int main( int argc, char *argv[])
			{
				//m = 4, n = 3, k = 6;
				//	k = 7;

				if (argc <= 2 ) {
					printf("./knight <linha_tabuleiro> <nro_cavalos>\n");
					return -1;
				}

				m = atoi(argv	[1]);
				n = m;
				k = atoi(argv [2]);

				/* Creation of a m*n board */
				char** board = new char*[m];
				for (int i = 0; i < m; i++) {
					board[i] = new char[n];
				}

				/* Make all the places are empty */
				makeBoard(board);
				//displayBoard(board);

				start = wtime();
				kkn(k, 0, 0, board);

				cout << endl
				<< "Total number of solutions : "
				<< count << endl;
				return 0;
			}

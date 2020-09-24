#define NO_PIECE			'_'
#define KNIGHT				'K'
#define QUEEN				'Q'
#define KNIGHT_ATTACK		'k'
#define QUEEN_ATTACK		'q'
#define DONT_PLACE_QUEEN	'*'
#define SKIP				'j'

typedef struct {
	char **board;
	int lin, col;
} chessboard;

__host__ void displayBoard(chessboard*);
__host__ void displayResult(chessboard*);
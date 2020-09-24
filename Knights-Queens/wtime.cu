#include <time.h>
#include <sys/time.h>

__host__ double wtime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + (double) t.tv_usec / 1000000;
}
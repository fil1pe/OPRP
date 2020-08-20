// ##########################################
//  
// Multiplicação de matrizes em CUDA
// Disciplina: OPRP001 - Programação Paralela
// Prof.: Mauricio Pillon
// Alunos: Filipe Ramos e Paula Campigotto
//  
// ##########################################

#include <cuda.h>
#include <stdio.h>
#include <math.h>

// Matriz Quadrada (nro_linhas = nro_colunas)
#define N 4 // Número de linhas 
            // Número de colunas

// GPU: Multiplicação das matrizes (a) e (b), resultado em (c)
__global__ void matMult (int *da, int *db, int *dc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  dc[i*N+j] = 0;

  for(int k=0; k<N; k++)
    dc[i*N+j] += da[i*N+k] * db[k*N+j];
}

// GPU: Imprime índices na matriz 
__global__ void printIndex (void) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;

   printf ("[%d][%d]=%d\t(x)\t%d\t%d\t%d\t(y)\t%d\t%d\t%d\n",i,j,(i*N+j), threadIdx.x, blockIdx.x, blockDim.x,threadIdx.y, blockIdx.y, blockDim.y);
}
   

// GPU: Inicializa os vetores (a), (b) e (c) na Memória Global 
__global__ void dirtyMem (int *da, int *db, int *dc) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   da[i] = -1;
   db[i] = -2;
   dc[i] = -3;
}

// CPU: Inicializa os vetores (a) e (b)
__host__ void initvet(int *host_a, int *host_b) {
  for (int i=0; i < N; i++) {
    for (int j=0; j < N; j++) {
       host_b[i*N+j] = (i+j)+((N-1)*i);
       host_a[i*N+j] = (N*N)-host_b[i*N+j];
    }
  }
}

// CPU: Imprime matriz
__host__ void printMat (int *mat){

	for (int j =0; j < N; j++)
	printf("\t(%d)", j);
	printf("\n");
	for (int i=0; i < N; i++) {
		printf("(%d)", i);
		for (int j=0; j < N; j++){
			printf("\t%d", mat[i*N+j]);
		}
		printf("\n");
	}
}
 

// CPU: função principal 
int main(int argc, char const *argv[]) {
  int *a, *b, *c;
  int *dev_a, *dev_b, *dev_c;
  int size;

  // Alocação de matriz quadrada
  size = N * N * sizeof(int);

  // Alocação de memória no host
  cudaMallocHost((void **) &a, size);
  cudaMallocHost((void **) &b, size);
  cudaMallocHost((void **) &c, size);
  
  // Alocação de memória na GPU para os vetores (a,b e c)
  cudaMalloc ((void **) &dev_a, size);
  cudaMalloc ((void **) &dev_b, size);
  cudaMalloc ((void **) &dev_c, size);

  // Atribui valores iniciais aos vetores em GPU
  dirtyMem<<<N, N>>>(dev_a, dev_b, dev_c);

  // Cópia GPU para CPU 
  cudaMemcpy (a, dev_a, size, cudaMemcpyDeviceToHost);
  cudaMemcpy (b, dev_b, size, cudaMemcpyDeviceToHost);
  cudaMemcpy (c, dev_c, size, cudaMemcpyDeviceToHost);

  // Impressão na tela dos valores dos vetores
  printf ("\t ### Valores Inicializados na GPU ###\n");
  printf ("\t ### Matriz (a) ### \n");
  printMat(a);
  printf ("\t ### Matriz (b) ### \n");
  printMat(b);
  printf ("\t ### Matriz (c) ### \n");
  printMat(c);

  // Inicialização dos vetores (a) e (b) no host
  initvet(a,b);

  // Cópia dos vetores gerados em CPU p/ memória da GPU
  cudaMemcpy (dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy (dev_b, b, size, cudaMemcpyHostToDevice);

  // Número de blocos e threads p/ dimensões (x,y)
  dim3 dimBlock (1, 1); 
  dim3 dimThreads(N, N); 

  // Imprime as posições acessadas pelo dimBlock e dimThreads
  printIndex<<< dimBlock, dimThreads>>>();

  // Execução do kernel matMult em GPU
  matMult<<< dimBlock, dimThreads>>>(dev_a, dev_b, dev_c);
  cudaDeviceSynchronize();

  // Cópia do vetor (c) da GPU (Memória Global) para CPU
  cudaMemcpy (c, dev_c, size, cudaMemcpyDeviceToHost);

  // Impressão na tela dos valores dos vetores
  printf ("\t ### Valores após processamento em GPU ###\n");
  printf ("\t ### Matriz (a) ### \n");
  printMat(a);
  printf ("\t ### Matriz (b) ### \n");
  printMat(b);
  printf ("\t ### Matriz (c) ### \n");
  printMat(c);

  // Libera a Memória Global (GPU)
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  // Libera a Memória Global (CPU)
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);

  return 0;
}

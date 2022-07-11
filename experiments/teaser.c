#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "cpu_timer.h"
#include <math.h>

/* Include polybench common header. */
// #include "polybench.h"

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
// #include "ludcmp.h"
#define N 5
typedef int DATA_TYPE ;

#define tsteps  5
#define n  20


static void init_array (
		 DATA_TYPE *A,
		 DATA_TYPE *B)
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
	    A[i*n+j] = i*(j+2) + 2;
	    B[i] = i*(j+3) + 3;
    }
}




// /* DCE code. Must scan the entire live-out data.
//    Can be used also to check the correctness of the output. */
static void print_array(DATA_TYPE *x)
{
  int i;

  for (i = 0; i < n; i++) 
  {
    printf("%d ", x[i]);
    // if (i % 20 == 0) fprintf (stderr, "\n");
  }

  printf("\n");
}


static void print(DATA_TYPE *x)
{
  int i,j;

  for (i = 0; i < n; i++) 
  {
    for(j=0 ; j< n ; j++)
      printf("%d ", x[i*n+j]);
    printf("\n");
  }

  printf("\n");
}


static
void kernel_jacobi_2d_imper(
			    DATA_TYPE *A,
			    DATA_TYPE *B,
          DATA_TYPE *X)
{
  int i,j,k;

  for (k = 0; k < n; k++)
  {
    for (j = k + 1; j < n; j++)
	    A[k*n+j] = (int) A[k*n+j] / (A[k*n+k]+100);
    for(i = k + 1; i < n; i++)
	    for (j = k + 1; j < n; j++)
	      A[i*n+j] = A[i*n+j] - A[i*n+k] * A[k*n+j];
  }

  // int X[n];
  X[0] = A[0];
  for(i = 1 ; i<n ; i++)
  {
    int w;
    for(j=0 ; j<i ; j++)
      w += X[j] * A[i*n+j];

    X[i] = B[i] - w*A[i*n+i-1];
  }

}



int main(int argc, char** argv)
{
  // int tmax = 5;
  // int nx = 10;
  // int ny = 10;

  DATA_TYPE *A = (int *) malloc(sizeof(int) * n * n);
  DATA_TYPE *B = (int *) malloc(sizeof(int) * n);
  DATA_TYPE *X = (int *) malloc(sizeof(int) * n);
  // DATA_TYPE hz[nx][ny];
  // DATA_TYPE fict[tmax];


  /* Initialize array(s). */
  init_array (A, B);


  cpu_timer t_test;
  timer_record_start(&t_test);

  kernel_jacobi_2d_imper(A, B, X);

  timer_record_stop(&t_test);
  timer_get_elapsed_time(&t_test,"time:",1);

  print(A);
  print(B);
  print_array(X);

  return 0;
}

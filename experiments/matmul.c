#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "cpu_timer.h"

#define N 2000
#define M 2000


typedef float llu;	

void g_1(llu *A, llu size)
{
    for (int i = 0; i < size; i++)
        // for (int j = 0; j < size; j++)
            A[i] = i+1;
}

void g_2(llu *A, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            A[i*size+j] = 0;
}

int compute_function_1(llu *C, llu *A, llu *B , int i, int *h)
{
    for(int j=1 ; j<N ; j++)
        // for(int k=1 ; k<N ; k++)
            C[i*N+j] += A[i*N+j] + B[i*N+j]; 

    // C[i] += C[i] + A[i] * B[i];

    return h[i]+1;
}

void task_func(llu *A, llu *B, llu *C, llu *D, llu *E, int *h, int *h1)
{
    int a;

    for (int i = 1; i < N; i++)
    {
        h[i] = compute_function_1(C,A,B,i,h);
    }

    for (int i = 1; i < N ; i++)
    {
        h[i] = h[i] + compute_function_1(E,C,D,i,h);
        
    }

    for (int i = 1; i < N; i++)
    {
        h[i] = compute_function_1(C,A,B,i,h);
    }

}

void print_func(llu *A)
{
    for (int i = 0; i < N ; i++)
    {
        for (int j = 0; j < N ; j++)
           printf("%f ", A[i*N+j]);
    //    printf("\n");
    }
    printf("\n");
    printf("===============================\n");
}

int main(int argc, char ** argv)
{

    llu *A = (llu *)malloc(sizeof(llu) * N * N);
    llu *B = (llu *)malloc(sizeof(llu) * N * N);
    llu *C = (llu *)malloc(sizeof(llu) * N * N);
    llu *D = (llu *)malloc(sizeof(llu) * N * N);
    llu *E = (llu *)malloc(sizeof(llu) * N * N);
    
    int *h = (int *) malloc(sizeof(int) * N);

    g_2(A, N);
    g_2(B, N);
    g_2(C, N);
    g_2(D, N);
    g_2(E, N);
    

    cpu_timer t_test;
    timer_record_start(&t_test);

    task_func(A,B,C,D,E,h);

    timer_record_stop(&t_test);
    timer_get_elapsed_time(&t_test,"time:",1);


    print_func(E);


    printf("Done\n\n");

    return 0;
}

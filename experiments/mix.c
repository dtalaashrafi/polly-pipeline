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

llu compute_function_1(llu *A, llu *B)
{
    llu s=1;
    for(int j=0 ; j<N ; j++)
        for(int k = 0 ; k<N ; k++)
            s = s + B[j] + A[k];

    return s;
}

void task_func(llu *A, llu *B, llu *C, llu *D, llu *E)
{
    

    for (int i = 1; i < N; i++)
        // for (int j = 0; j < N-1; j++)
    {
        C[i] = C[i] + compute_function_1(A,B);
    }

    for (int i = 1; i < N ; i++)
    {
        D[i] = D[i-1] * C[i];
        E[i] = E[i-1] * D[i];
    }

}

void print_func(llu *A, llu *B)
{
    for (int i = 0; i < N ; i++)
    {
        // for (int j = 0; j < N ; j++)
           printf("%f ", A[i]);
    //    printf("\n");
    }
    printf("\n");
    printf("===============================\n");

    for (int i = 0; i < M ; i++)
    {
        // for (int j = 0; j < M ; j++)
           printf("%f ", B[i]);
    //    printf("\n");
    }
    printf("\n");
    printf("===============================\n");
}

int main(int argc, char ** argv)
{

    llu *A = (llu *)malloc(sizeof(llu) * N);
    llu *B = (llu *)malloc(sizeof(llu) * N);
    llu *C = (llu *)malloc(sizeof(llu) * N);
    llu *D = (llu *)malloc(sizeof(llu) * N);
    llu *E = (llu *)malloc(sizeof(llu) * N);
    
    
    g_1(A, N);
    g_1(B, N);
    g_1(C, N);
    g_1(D, N);
    g_1(E, N);
    

    cpu_timer t_test;
    timer_record_start(&t_test);

    task_func(A,B,C,D,E);

    timer_record_stop(&t_test);
    timer_get_elapsed_time(&t_test,"time:",1);

    print_func(A,B);
    print_func(C,D);
    print_func(D,E);


    printf("Done\n\n");

    return 0;
}

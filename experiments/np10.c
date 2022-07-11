// #include "computation_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include "cpu_timer.h"

// #define N 20
// #define M 20
// #define Q 20
// #define SIZE 10
// #define num1 2
// #define num2 2
// #define num3 2

#define N 100
#define M 100
#define Q 100
#define SIZE 100
#define num1 1
#define num2 1
#define num3 2
#define num4 2
#define DONOTCHECK 

typedef struct gmp_data
{
    mpz_t *data;
    int status;
}gmp_data;


void initialize(gmp_data *a , int p)
{
    mpz_t n;
    mpz_init(n);
    mpz_ui_pow_ui(n, 2, p); //n = 2^p-1
    mpz_sub_ui(n, n, 1);

    for(int i=0 ; i<SIZE ; i++)
    {
        mpz_set(a->data[i], n);
	mpz_mul_ui(a->data[i], a->data[i], i);
    }

    a->status = 0;
}

void print(gmp_data *a)
{
    // printf("[");
    for(int k=0 ; k<SIZE ; k++)
        gmp_printf("%Zd ", a->data[k]);
    // printf("]");
}

//  res = mA
gmp_data *compute_function_1(gmp_data *res, gmp_data *a, mpz_t m)
{
    for(int i=0 ; i<SIZE ; i++)
    {
        mpz_mul(res->data[i], a->data[i], m);
        for(int k=0 ; k<num1 ; k++)
        {
            mpz_add_ui(res->data[i] , res->data[i] , 1);
            mpz_nextprime(res->data[i] , res->data[i]);
        }
    }

    return res;
}

gmp_data *compute_function_2(gmp_data *res, gmp_data *a, gmp_data *b, mpz_t m)
{
    for(int i=0 ; i<SIZE ; i++)
    {
        mpz_add(res->data[i], a->data[i], b->data[i]);
	mpz_mul(res->data[i], res->data[i], m);
        for(int k=0 ; k<num2 ; k++)
        {
            mpz_add_ui(res->data[i] , res->data[i] , 1);
            mpz_nextprime(res->data[i] , res->data[i]);
        }
    }
    return res;
}


gmp_data *compute_function_3(gmp_data *res, gmp_data *a, gmp_data *b, gmp_data *c)
{
    for(int i=0 ; i<SIZE ; i++)
    {
        mpz_add(res->data[i], a->data[i], b->data[i]);
        mpz_add(res->data[i], res->data[i], c->data[i]);
        for(int k=0 ; k<num3 ; k++)
        {
            mpz_add_ui(res->data[i] , res->data[i] , 1);
            mpz_nextprime(res->data[i] , res->data[i]);
        }
        
    }
    return res;
}

gmp_data *compute_function_4(gmp_data *res, gmp_data *a, gmp_data *b, gmp_data *c, gmp_data *d)
{
    for(int i=0 ; i<SIZE ; i++)
    {
        mpz_add(res->data[i], a->data[i], b->data[i]);
        mpz_add(res->data[i], res->data[i], c->data[i]);
        mpz_add(res->data[i], res->data[i], d->data[i]);
        for(int k=0 ; k<num4 ; k++)
        {
            mpz_add_ui(res->data[i] , res->data[i] , 1);
            mpz_nextprime(res->data[i] , res->data[i]);
        }
        
    }
    return res;
}

gmp_data *compute_function_add_4(gmp_data *res, gmp_data *a, gmp_data *b, gmp_data *c, gmp_data *d, int num)
{
    for(int i=0 ; i<SIZE ; i++)
    {
        mpz_add(res->data[i], a->data[i], b->data[i]);
        mpz_add(res->data[i], res->data[i], c->data[i]);
        mpz_add(res->data[i], res->data[i], d->data[i]);
        for(int k=0 ; k<num ; k++)
        {
            mpz_add_ui(res->data[i] , res->data[i] , 1);
            mpz_nextprime(res->data[i] , res->data[i]);
        }
        
    }
    return res;
}

gmp_data *compute_function_mult(gmp_data *res, gmp_data *a, gmp_data *b)
{
    mpz_t temp;
    mpz_init(temp);
    // for(int i=0 ; i<2*SIZE ; i++)
    //     mpz_set_ui(res->data[i],0);

    for(int i=0 ; i<SIZE/2 ; i++)
        for(int j=0 ; j<SIZE/2 ; j++)
        {
            mpz_mul(temp, a->data[i], b->data[j]);
            mpz_add(res->data[i+j], res->data[i+j], temp);
        }
    return res;
}



void task_func(gmp_data **A , gmp_data **B, gmp_data **C, gmp_data **D , mpz_t m)
{

    for(int i=0 ; i<N-1 ; i++)
        for(int j=1 ; j<N-1 ; j++)
            A[i*N+j] = compute_function_2(A[i*N+j], A[i*N+(j-1)], A[(i+1)*N+(j+1)],m);

    for(int i=0 ; i<M/2-1 ; i++)
        for(int j=1 ; j<M/2-1 ; j++)
            B[i*N+j] = compute_function_3(B[i*M+j], B[i*M+(j-1)], B[(i+1)*M+(j+1)],A[(i+j)*N+(j)]);

    for(int i=0 ; i<Q-1 ; i++)
        for(int j=1 ; j<Q-1 ; j++)
            C[i*N+j] = compute_function_4(C[i*M+j], C[i*M+(j-1)], C[(i+1)*M+(j+1)], C[(i)*N+(j)], A[(i)*N+(j)]); 
	
   for(int i=0 ; i<Q-1 ; i++)
       for(int j=1 ; j<Q-1 ; j++)
          D[i*N+j] = compute_function_4(D[i*M+j], D[i*M+(j-1)], D[(i+1)*M+(j+1)], D[(i)*N+j], B[(i)*N+(j)]);	       

}

int main(int argc, char ** argv)
{

    gmp_data **A = (gmp_data**) malloc(sizeof(gmp_data*)*N*N);
    for(int i=0 ; i<N ; i++)
        for(int j=0 ; j<N ; j++)
        {
            A[i*N+j] = (gmp_data*) malloc(sizeof(gmp_data));
            A[i*N+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }
    for(int i=0 ; i<N ; i++)
        for(int j=0 ; j<N ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(A[i*N+j]->data[k]);
//////////////////////////////////////////////////////////////////////////
    gmp_data **B = (gmp_data**) malloc(sizeof(gmp_data*)*M*M);
    for(int i=0 ; i<M ; i++)
        for(int j=0 ; j<M ; j++)
        {
            B[i*M+j] = (gmp_data*) malloc(sizeof(gmp_data));
            B[i*M+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }
    for(int i=0 ; i<M ; i++)
        for(int j=0 ; j<M ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(B[i*M+j]->data[k]);
//////////////////////////////////////////////////////////////////////////    
    gmp_data **C = (gmp_data**) malloc(sizeof(gmp_data*)*Q*Q);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
        {
            C[i*Q+j] = (gmp_data*) malloc(sizeof(gmp_data));
            C[i*Q+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(C[i*Q+j]->data[k]);
//////////////////////////////////////////////////////////////////////////
    gmp_data **D = (gmp_data**) malloc(sizeof(gmp_data*)*Q*Q);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
        {
            D[i*Q+j] = (gmp_data*) malloc(sizeof(gmp_data));
            D[i*Q+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(D[i*Q+j]->data[k]);
//////////////////////////////////////////////////////////////////////////
    gmp_data **E = (gmp_data**) malloc(sizeof(gmp_data*)*Q*Q);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
        {
            E[i*Q+j] = (gmp_data*) malloc(sizeof(gmp_data));
            E[i*Q+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(E[i*Q+j]->data[k]);
//////////////////////////////////////////////////////////////////////////
    gmp_data **F = (gmp_data**) malloc(sizeof(gmp_data*)*Q*Q);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
        {
            F[i*Q+j] = (gmp_data*) malloc(sizeof(gmp_data));
            F[i*Q+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(F[i*Q+j]->data[k]);
//////////////////////////////////////////////////////////////////////////
    gmp_data **G = (gmp_data**) malloc(sizeof(gmp_data*)*Q*Q);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
        {
            G[i*Q+j] = (gmp_data*) malloc(sizeof(gmp_data));
            G[i*Q+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(G[i*Q+j]->data[k]);
//////////////////////////////////////////////////////////////////////////
    gmp_data **H = (gmp_data**) malloc(sizeof(gmp_data*)*Q*Q);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
        {
            H[i*Q+j] = (gmp_data*) malloc(sizeof(gmp_data));
            H[i*Q+j]->data = (mpz_t*) malloc(sizeof(mpz_t)*SIZE);
        }

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_init(H[i*Q+j]->data[k]);
//////////////////////////////////////////////////////////////////////////

    for(int i=0 ; i<N ; i++)
        for(int j=0 ; j<N ; j++)
            initialize(A[i*N+j], 1);
    for(int i=0 ; i<M ; i++)
        for(int j=0 ; j<M ; j++)
            initialize(B[i*M+j], 14);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            initialize(C[i*Q+j], 0);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            initialize(D[i*Q+j], 1);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            initialize(E[i*Q+j], 1);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            initialize(F[i*Q+j], 1);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            initialize(G[i*Q+j], 1);
    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            initialize(H[i*Q+j], 1);
//////////////////////////////////////////////////////////////////////////

    for(int i=0 ; i<N ; i++)
        for(int j=0 ; j<N ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_add_ui(A[i*N+j]->data[k], A[i*N+j]->data[k], j+i);
                
    for(int i=0 ; i<M ; i++)
        for(int j=0 ; j<M ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_add_ui(B[i*M+j]->data[k], B[i*M+j]->data[k], j+i);

    mpz_t m;
    mpz_init(m);
    mpz_set_ui(m,7);



    cpu_timer t_test;
    timer_record_start(&t_test);

    task_func(A,B,C,D,m);

    timer_record_stop(&t_test);
    timer_get_elapsed_time(&t_test,"time:",1);

#ifdef CHECKED
    for(int i=0 ; i<N ; i++)
    {
        for(int j=0 ; j<N ; j++)
        {
            gmp_printf("%Zd ", A[i*N+j]->data[0]);
            // printf("  ");
        }
        printf("\n");
    }

    for(int i=0 ; i<M ; i++)
    {
        for(int j=0 ; j<M ; j++)
        {
            print(B[i*M+j]);
            // printf("  ");
        }
        printf("\n");
    }

    for(int i=0 ; i<Q ; i++)
    {
        for(int j=0 ; j<Q ; j++)
        {
            print(C[i*Q+j]);
            // printf("  ");
        }
        printf("\n");
    }

    for(int i=0 ; i<Q ; i++)
    {
        for(int j=0 ; j<Q ; j++)
        {
            print(D[i*Q+j]);
            // printf("  ");
        }
        printf("\n");
    }

    for(int i=0 ; i<Q ; i++)
    {
        for(int j=0 ; j<Q ; j++)
        {
            print(E[i*Q+j]);
            // printf("  ");
        }
        printf("\n");
    }
    for(int i=0 ; i<Q ; i++)
    {
        for(int j=0 ; j<Q ; j++)
        {
            print(F[i*Q+j]);
            // printf("  ");
        }
        printf("\n");
    }
    for(int i=0 ; i<Q ; i++)
    {
        for(int j=0 ; j<Q ; j++)
        {
            print(G[i*Q+j]);
            // printf("  ");
        }
        printf("\n");
    }
    for(int i=0 ; i<Q ; i++)
    {
        for(int j=0 ; j<Q ; j++)
        {
            print(H[i*Q+j]);
            // printf("  ");
        }
        printf("\n");
    }
#endif

    for(int i=0 ; i<N ; i++)
        for(int j=0 ; j<N ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(A[i*N+j]->data[k]);

    for(int i=0 ; i<N ; i++)
        for(int j=0 ; j<N ; j++)
           free(A[i*N+j]->data);
    free(A);

    for(int i=0 ; i<M ; i++)
        for(int j=0 ; j<M ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(B[i*M+j]->data[k]);

    for(int i=0 ; i<M ; i++)
        for(int j=0 ; j<M ; j++)
           free(B[i*M+j]->data);
    free(B);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(C[i*M+j]->data[k]);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
           free(C[i*M+j]->data);
    free(C);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(D[i*M+j]->data[k]);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
           free(D[i*M+j]->data);
    free(D);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(E[i*M+j]->data[k]);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
           free(E[i*M+j]->data);
    free(E);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(F[i*M+j]->data[k]);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
           free(F[i*M+j]->data);
    free(F);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(G[i*M+j]->data[k]);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
           free(G[i*M+j]->data);
    free(G);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
            for(int k=0 ; k<SIZE ; k++)
                mpz_clear(H[i*M+j]->data[k]);

    for(int i=0 ; i<Q ; i++)
        for(int j=0 ; j<Q ; j++)
           free(H[i*M+j]->data);
    free(H);


    return 0;
}

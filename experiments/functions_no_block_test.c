#include<stdio.h>
#include<stdlib.h>
#include <omp.h>

int count = 0;

int *depend_arr = NULL ;
int func_count[4];


void make_task(void (*f) (void *), void *input, long int out_depend, int out_index, long int *in_depend, 
                    int *in_index, int input_size, int depend_num)
{

    int writer_number = 1;
    if(out_depend < 0) 
        return;

    count++;
    void *new_input = malloc(input_size);
    memcpy(new_input, input, input_size);

    int *self = (int *) f;
   
    func_count[out_index]++;

    if(in_index[0] == -1)
    {
        if(func_count[out_index] == 0)
        {
            #pragma omp task depend(out:depend_arr[writer_number * out_depend + out_index]) \
                             depend(out: self[func_count[out_index]]) \
                             
            {    
                f(new_input);
                free(new_input);
            }
        }
        else
        {
            #pragma omp task depend(out:depend_arr[writer_number * out_depend + out_index]) \
                             depend(out: self[func_count[out_index]]) \
                             depend(in: self[func_count[out_index]-1]) \
                             
            {   
                f(new_input);
                free(new_input);
            }
        }
    }

    else
    {
        if(func_count[out_index] <= 0) 
        {
            #pragma omp task depend(out:depend_arr[writer_number * out_depend + out_index]) \
                             depend(iterator (i = 0:depend_num), in: depend_arr[writer_number*in_depend[i] + in_index[i]]) \
                             depend(out: self[func_count[out_index]]) \
                             
            {   
                f(new_input);
                free(new_input);
            }

            return;
        }
        {
            #pragma omp task depend(out:depend_arr[writer_number * out_depend + out_index]) \
                                depend(iterator (i = 0:depend_num), in: depend_arr[writer_number*in_depend[i] + in_index[i]]) \
                                depend(in: self[func_count[out_index]-1]) \
                                depend(out: self[func_count[out_index]])                             
            {  
                f(new_input);
                free(new_input);
            }
        }
    }

}

/////////////////////////////////////////////////////
void make_parallel_region(void (*g) (void *), void *input)
{
    func_count[0] = -1;
    func_count[1] = -1;
    func_count[2] = -1;
    func_count[3] = -1;
    omp_set_num_threads(2) ;
    #pragma omp parallel
    {
        #pragma omp master
        {
            g(input); //task_func
        }
    }
}


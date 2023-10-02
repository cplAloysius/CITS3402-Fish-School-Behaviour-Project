#include<stdio.h>
#include<omp.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
FILE *fp1, *fp2, *fopen();
int process_id, number_of_processes;
2
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD,&process_id);
MPI_Comm_size(MPI_COMM_WORLD,&number_of_processes);
#pragma omp parallel
{
printf("Hello world, I am process %d among %d processes
and thread_id %d among %d threads\n",process_id,number_of_processes,
omp_get_thread_num(),omp_get_num_threads());
if ((process_id==0)&&(omp_get_thread_num()==0))
{
fp1 = fopen("out1.txt","w+");
fp2 = fopen("out2.txt","w+");
fprintf(fp1,"I am thread %d of process %d\n",
omp_get_thread_num(),process_id);
fprintf(fp2,"I am thread %d of process %d\n",
omp_get_thread_num(),process_id);
}
}
MPI_Finalize();
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define NUM_FISH 100 // Number of fish in the school
#define W_INITIAL 10.0   // Initial weight of each fish
#define MASTER 0

// Structure to represent a fish
typedef struct
{
    double x, y; // Coordinates
    double weight;
} Fish;

void write_to_file(Fish *school, int no_fish, const char *filename) {
    FILE *file = fopen(filename, "w");
    for (int i = 0; i < no_fish; i++) {
        fprintf(file, "%.2lf %.2lf %.2lf\n", school[i].x, school[i].y, school[i].weight);
    }
    fclose(file);
}

int main()
{
    FILE *fp1, *fp2, *fopen();

    int node_id, no_nodes;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &no_nodes);

    Fish *school_send = NULL;

    if (node_id == MASTER) {
        school_send = (Fish *)malloc(NUM_FISH * sizeof(Fish));
        for (int i = 0; i < NUM_FISH; i++)
        {
            school_send[i].x = (double)(rand() % 201 - 100); // Random x-coordinate between -100 and 100
            school_send[i].y = (double)(rand() % 201 - 100); // Random y-coordinate between -100 and 100
            school_send[i].weight = W_INITIAL;
        }
        write_to_file(school_send, NUM_FISH, "file1");
	}
    
    int fish_nodes = NUM_FISH / no_nodes;
    Fish *school_rec = (Fish *)malloc(fish_nodes * sizeof(Fish));

    MPI_Scatter(school_send, fish_nodes, MPI_DOUBLE, school_rec, fish_nodes, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    Fish *school_master = NULL;
    if (node_id == MASTER) {
        school_master = (Fish *)malloc(NUM_FISH * sizeof(Fish));
    }

    MPI_Gather(school_rec, fish_nodes, MPI_DOUBLE, school_master, fish_nodes, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if (node_id == MASTER) {
        write_to_file(school_master, NUM_FISH, "file2");
    }

    if (node_id == MASTER) {
        free(school_send);
        free(school_master);
    }
    free(school_rec);

    MPI_Finalize();
    return 0;



    

    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    // MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    // // Free dynamically allocated memory
    // free(school);

    // return 0;
}

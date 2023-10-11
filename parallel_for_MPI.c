#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define NUM_FISH 100 // Number of fish in the school
#define NUM_FIELDS 4
#define MASTER 0
#define NUM_STEPS 10   // Number of simulation steps
#define W_INITIAL 10.0   // Initial weight of each fish
#define W_MAX (2 * W_INITIAL)

// Structure to represent a fish
typedef struct
{
    double x, y; // Coordinates
    double delta_fi;
    double weight;
} Fish;

// Function to calculate the objective function for a fish
double calculateObjective(Fish *fish)
{
    return sqrt(fish->x * fish->x + fish->y * fish->y);
}

MPI_Datatype create_mpi_struct() {
    MPI_Datatype MPI_SCHOOL_STRUCT;
    MPI_Datatype types[NUM_FIELDS];
    MPI_Aint offsets[NUM_FIELDS];
    int blocklengths[NUM_FIELDS];

    MPI_Type_contiguous(1, MPI_DOUBLE, &types[0]);
    MPI_Type_contiguous(1, MPI_DOUBLE, &types[1]);
    MPI_Type_contiguous(1, MPI_DOUBLE, &types[2]);
    MPI_Type_contiguous(1, MPI_DOUBLE, &types[3]);

    offsets[0] = 0;
    offsets[1] = sizeof(double);
    offsets[2] = 2 * sizeof(double);
    offsets[3] = 3 * sizeof(double);

    blocklengths[0] = 1;
    blocklengths[1] = 1;
    blocklengths[2] = 1;
    blocklengths[3] = 1;

    MPI_Type_create_struct(NUM_FIELDS, blocklengths, offsets, types, &MPI_SCHOOL_STRUCT);
    MPI_Type_commit(&MPI_SCHOOL_STRUCT);

    return MPI_SCHOOL_STRUCT;
}

int main(int argc, char* argv[])
{
    printf("Parallel program\n");
    // Initialize random number generator
    srand(time(NULL));

    int process_id, num_processes;
    MPI_Init(0, 0);
    const MPI_Datatype MPI_SCHOOL_STRUCT = create_mpi_struct();

    
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int num_fish_per_process = NUM_FISH / num_processes;
    int num_elements = num_fish_per_process * NUM_FIELDS;

    Fish *school_send = NULL;
    if (process_id == MASTER) {
        school_send = (Fish*)malloc(NUM_FISH * sizeof(Fish));
        if (school_send == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        for (int i = 0; i < NUM_FISH; i++)
        {
            school_send[i].x = (double)(rand() % 201 - 100); // Random x-coordinate between -100 and 100
            school_send[i].y = (double)(rand() % 201 - 100); // Random y-coordinate between -100 and 100
            school_send[i].weight = W_INITIAL;
        }
    }

    Fish* school_rec = (Fish*)malloc(num_fish_per_process * sizeof(Fish));
    if (school_rec == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Scatter(school_send, num_elements, MPI_SCHOOL_STRUCT, school_rec, num_elements, MPI_SCHOOL_STRUCT, MASTER, MPI_COMM_WORLD);

    // Start timing
    double start_time = omp_get_wtime();

    // Simulation loop
    for (int step = 0; step < NUM_STEPS; step++)
    {
        double delta_fi_max = 0.0, bari_num = 0.0, bari_denom = 0.0;
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < num_fish_per_process; i++)
            {
                // Calculate change in objective function
                double old_obj = calculateObjective(&school_rec[i]);
                double new_x = school_rec[i].x + (double)(rand() % 21 - 10) / 100.0;
                double new_y = school_rec[i].y + (double)(rand() % 21 - 10) / 100.0;
                school_rec[i].x = new_x;
                school_rec[i].y = new_y;
                double new_obj = calculateObjective(&school_rec[i]);
                double delta_fi = new_obj - old_obj;
                school_rec[i].delta_fi = delta_fi;
            }

#pragma omp for reduction(max : delta_fi_max)
            for (int i = 0; i < num_fish_per_process; i++)
            {
                if (school_rec[i].delta_fi > delta_fi_max)
                {
                    delta_fi_max = school_rec[i].delta_fi;
                }
            }
        }
            double global_delta_fi_max;
            MPI_Reduce(&delta_fi_max, &global_delta_fi_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);
            MPI_Bcast(&global_delta_fi_max, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
#pragma omp parallel
        {
#pragma omp for reduction(+ : bari_num, bari_denom)
            for (int i = 0; i < num_fish_per_process; i++)
            {
                // Update fish weight
                school_rec[i].weight = fmin(W_MAX, school_rec[i].weight + school_rec[i].delta_fi / delta_fi_max);
                school_rec[i].weight = fmax(0, school_rec[i].weight);

                bari_num += calculateObjective(&school_rec[i]) * school_rec[i].weight;
                bari_denom += calculateObjective(&school_rec[i]);
            }
        }
        double global_bari_num;
        double global_bari_denom;

        MPI_Reduce(&bari_num, &global_bari_num, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&bari_denom, &global_bari_denom, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

        //double bari = bari_num / bari_denom;
        if (process_id == MASTER) {
            double bari = global_bari_num / global_bari_denom;
            if (bari <= 0) {
                fprintf(stderr, "Error: barycentre is non-positive.\n");
                MPI_Finalize();
                return EXIT_FAILURE;
            }

            //if (step % (NUM_STEPS / 20) == 0)
            printf("Step %d - Barycenter: %lf\n", step + 1, bari);
        }
    }

    // Calculate and print elapsed time
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Elapsed Time: %lf seconds\n", elapsed_time);

    // Free dynamically allocated memory
    free(school_send);
    free(school_rec);
    MPI_Finalize();

    return 0;
}

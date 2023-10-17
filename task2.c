#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#define NUM_FISH 1000000
#define NUM_STEPS 3000 
#define MASTER 0
#define FIELDS 4
#define W_INITIAL 10
#define W_MAX (2 * W_INITIAL)

// Define the Fish structure
typedef struct {
    double x;
    double y;
    double delta_fi;
    double weight;
} Fish;

MPI_Datatype create_mpi_struct() {
    MPI_Datatype MPI_SCHOOL_STRUCT;
    MPI_Datatype types[FIELDS];
    int blocklengths[FIELDS];
    MPI_Aint offsets[FIELDS];

    for (int i = 0; i < FIELDS; i++) {
        types[i] = MPI_DOUBLE;
        blocklengths[i] = 1;
    }

    offsets[0] = offsetof(Fish, x);
    offsets[1] = offsetof(Fish, y);
    offsets[2] = offsetof(Fish, delta_fi);
    offsets[3] = offsetof(Fish, weight);

    MPI_Type_create_struct(FIELDS, blocklengths, offsets, types, &MPI_SCHOOL_STRUCT);
    MPI_Type_commit(&MPI_SCHOOL_STRUCT);

    return MPI_SCHOOL_STRUCT;
}

double calculateObjective(Fish *fish)
{
    return sqrt(fish->x * fish->x + fish->y * fish->y);
}

int main() {
    int node_id, num_nodes;
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

    Fish *school_send = NULL;
    Fish *school_rec = NULL;
    const MPI_Datatype MPI_SCHOOL_STRUCT = create_mpi_struct();

    if (node_id == MASTER) {
        school_send = (Fish *)malloc(NUM_FISH * sizeof(Fish));
        if (school_send == NULL) {
            perror("Memory allocation error");
            MPI_Finalize();
            return 1;
        }

        for (int i = 0; i < NUM_FISH; i++) {
            // Initialize send_buf with fish data (not shown in the C code)
            school_send[i].x = (double)(rand() % 201 - 100); // Random x-coordinate between -100 and 100
            school_send[i].y = (double)(rand() % 201 - 100); // Random y-coordinate between -100 and 100
            school_send[i].weight = W_INITIAL;
        }
    }

    const int fish_per_node = NUM_FISH / num_nodes;
    school_rec = (Fish *)malloc(fish_per_node * sizeof(Fish));

    MPI_Scatter(school_send, fish_per_node, MPI_SCHOOL_STRUCT, school_rec, fish_per_node, MPI_SCHOOL_STRUCT, MASTER, MPI_COMM_WORLD);

    double start_time = omp_get_wtime();

    for (int steps = 0; steps < NUM_STEPS; steps++) {
        double delta_fi_max = 0.0;
        double bari_num = 0.0;
        double bari_denom = 0.0;

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < fish_per_node; i++)
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
            for (int i = 0; i < fish_per_node; i++)
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
            for (int i = 0; i < fish_per_node; i++)
            {
                // Update fish weight
                school_rec[i].weight = fmin(W_MAX, school_rec[i].weight + school_rec[i].delta_fi / global_delta_fi_max);
                school_rec[i].weight = fmax(0, school_rec[i].weight);

                bari_num += calculateObjective(&school_rec[i]) * school_rec[i].weight;
                bari_denom += calculateObjective(&school_rec[i]);
            }
        }

        double global_num;
        double global_denom;
        MPI_Reduce(&bari_num, &global_num, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&bari_denom, &global_denom, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

        if (node_id == MASTER) {
            double barycentre = global_num / global_denom;
            if (barycentre <= 0) {
                fprintf(stderr, "Exception: barycentre is less than or equal to 0\n");
                MPI_Finalize();
                return 1;
            }

            printf("Step %d - Barycenter: %lf\n", steps + 1, barycentre);
        }
    }

    if (node_id == MASTER) {
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        printf("Elapsed Time: %lf seconds\n", elapsed_time);
    }

    free(school_send);
    free(school_rec);
    MPI_Finalize();
    return 0;
}

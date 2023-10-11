#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#define NUM_FISH 100000
#define NUM_STEPS 300 
#define MASTER 0
#define NUM_FIELDS 4
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
    MPI_Datatype MPI_FISH;
    MPI_Datatype types[NUM_FIELDS];
    int blocklengths[NUM_FIELDS];
    MPI_Aint offsets[NUM_FIELDS];

    for (int i = 0; i < NUM_FIELDS; i++) {
        types[i] = MPI_DOUBLE;
        blocklengths[i] = 1;
    }

    offsets[0] = offsetof(Fish, x);
    offsets[1] = offsetof(Fish, y);
    offsets[2] = offsetof(Fish, delta_fi);
    offsets[3] = offsetof(Fish, weight);

    MPI_Type_create_struct(NUM_FIELDS, blocklengths, offsets, types, &MPI_FISH);
    MPI_Type_commit(&MPI_FISH);

    return MPI_FISH;
}

double calculateObjective(Fish *fish)
{
    return sqrt(fish->x * fish->x + fish->y * fish->y);
}

int main() {
    int process_id, num_processes;
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    Fish *send_buf = NULL;
    Fish *recv_buf = NULL;
    const MPI_Datatype MPI_FISH = create_mpi_struct();

    if (process_id == MASTER) {
        send_buf = (Fish *)malloc(NUM_FISH * sizeof(Fish));
        if (send_buf == NULL) {
            perror("Memory allocation error");
            MPI_Finalize();
            return 1;
        }

        for (int i = 0; i < NUM_FISH; i++) {
            // Initialize send_buf with fish data (not shown in the C code)
            send_buf[i].x = (double)(rand() % 201 - 100); // Random x-coordinate between -100 and 100
            send_buf[i].y = (double)(rand() % 201 - 100); // Random y-coordinate between -100 and 100
            send_buf[i].weight = W_INITIAL;
        }
    }

    const int num_fish_per_process = NUM_FISH / num_processes;
    recv_buf = (Fish *)malloc(num_fish_per_process * sizeof(Fish));

    MPI_Scatter(send_buf, num_fish_per_process, MPI_FISH, recv_buf, num_fish_per_process, MPI_FISH, MASTER, MPI_COMM_WORLD);

    double start_time = omp_get_wtime();

    for (int steps = 0; steps < NUM_STEPS; steps++) {
        double max_difference = 0.0;
        double local_numerator = 0.0;
        double local_denominator = 0.0;

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < num_fish_per_process; i++)
            {
                // Calculate change in objective function
                double old_obj = calculateObjective(&recv_buf[i]);
                double new_x = recv_buf[i].x + (double)(rand() % 21 - 10) / 100.0;
                double new_y = recv_buf[i].y + (double)(rand() % 21 - 10) / 100.0;
                recv_buf[i].x = new_x;
                recv_buf[i].y = new_y;
                double new_obj = calculateObjective(&recv_buf[i]);
                double delta_fi = new_obj - old_obj;
                recv_buf[i].delta_fi = delta_fi;
            }

            #pragma omp for reduction(max : max_difference)
            for (int i = 0; i < num_fish_per_process; i++)
            {
                if (recv_buf[i].delta_fi > max_difference)
                {
                    max_difference = recv_buf[i].delta_fi;
                }
            }
        }

        double global_max_difference;
        MPI_Reduce(&max_difference, &global_max_difference, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);
        MPI_Bcast(&global_max_difference, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        #pragma omp parallel
        {
            #pragma omp for reduction(+ : local_numerator, local_denominator)
            for (int i = 0; i < num_fish_per_process; i++)
            {
                // Update fish weight
                recv_buf[i].weight = fmin(W_MAX, recv_buf[i].weight + recv_buf[i].delta_fi / global_max_difference);
                recv_buf[i].weight = fmax(0, recv_buf[i].weight);

                local_numerator += calculateObjective(&recv_buf[i]) * recv_buf[i].weight;
                local_denominator += calculateObjective(&recv_buf[i]);
            }
        }

        double global_numerator;
        double global_denominator;
        MPI_Reduce(&local_numerator, &global_numerator, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&local_denominator, &global_denominator, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

        if (process_id == MASTER) {
            double barycentre = global_numerator / global_denominator;
            if (barycentre <= 0) {
                fprintf(stderr, "Exception: barycentre is less than or equal to 0\n");
                MPI_Finalize();
                return 1;
            }

            printf("%lf\n", barycentre);
        }
    }

    if (process_id == MASTER) {
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        printf("Elapsed Time: %lf seconds\n", elapsed_time);
    }

    free(send_buf);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}

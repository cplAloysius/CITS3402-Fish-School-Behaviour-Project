#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define NUM_FISH 1000000// Number of fish in the school
#define NUM_STEPS 30   // Number of simulation steps
#define W_INITIAL 100.0 // Initial weight of each fish
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

int main()
{
    // Initialize random number generator
    srand(time(NULL));

    // Create an array of fish dynamically
    Fish *school = (Fish *)malloc(NUM_FISH * sizeof(Fish));
    if (school == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < NUM_FISH; i++)
    {
        school[i].x = (double)(rand() % 201 - 100); // Random x-coordinate between -100 and 100
        school[i].y = (double)(rand() % 201 - 100); // Random y-coordinate between -100 and 100
        school[i].weight = W_INITIAL;
    }

    // Start timing
    double start_time = omp_get_wtime();

    // Simulation loop
    for (int step = 0; step < NUM_STEPS; step++)
    {
        double delta_fi_max = 0.0;
       
        #pragma omp parallel
        {
     
            double thread_max = 0.0;
        #pragma omp for

        for (int i = 0; i < NUM_FISH; i++)
        {

            // Calculate change in objective function
            double old_obj = calculateObjective(&school[i]);
            double new_x = school[i].x + (double)(rand() % 21 - 10) / 100.0;
            double new_y = school[i].y + (double)(rand() % 21 - 10) / 100.0;
            school[i].x = new_x;
            school[i].y = new_y;
            double new_obj = calculateObjective(&school[i]);
            double delta_fi = fabs(new_obj - old_obj);
            school[i].delta_fi = delta_fi;

            if (delta_fi > thread_max)
            {
                thread_max = delta_fi;
            }
        }
        #pragma omp critical
        {
            if (thread_max > delta_fi_max)
            {
                delta_fi_max = thread_max;
            }
        }
        }


        double bari_num = 0.0, bari_denom = 0.0, bari = 0.0;
        for (int i = 0; i < NUM_FISH; i++)
        {
            // Update fish weight
            school[i].weight = fmin(W_MAX, school[i].weight + school[i].delta_fi / delta_fi_max);

            bari_num += calculateObjective(&school[i]) * school[i].weight;
            bari_denom += calculateObjective(&school[i]);
        }
        bari = bari_num / bari_denom;

        printf("Step %d - Barycenter: %lf\n", step + 1, bari);

        // double bari_x = 0.0, bari_y = 0.0, total_weight = 0.0;
        // for (int i = 0; i < NUM_FISH; i++)
        // {
        //     bari_x += school[i].x * school[i].weight;
        //     bari_y += school[i].y * school[i].weight;
        //     total_weight += school[i].weight;
        // }
        // bari_x /= total_weight;
        // bari_y /= total_weight;

        // // Display barycenter
        // printf("Step %d - Barycenter: (%lf, %lf)\n", step + 1, bari_x, bari_y);
    }

    // Calculate and print elapsed time
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Elapsed Time: %lf seconds\n", elapsed_time);

    // Free dynamically allocated memory
    free(school);

    return 0;
}

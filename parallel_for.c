#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define NUM_FISH 1000000 // Number of fish in the school
#define NUM_STEPS 3000   // Number of simulation steps
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

int main()
{
    printf("Parallel program\n");
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
        double delta_fi_max = 0.0, bari_num = 0.0, bari_denom = 0.0;
#pragma omp parallel
        {
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
                double delta_fi = new_obj - old_obj;
                school[i].delta_fi = delta_fi;
            }

#pragma omp for reduction(max : delta_fi_max)
            for (int i = 0; i < NUM_FISH; i++)
            {
                if (school[i].delta_fi > delta_fi_max)
                {
                    delta_fi_max = school[i].delta_fi;
                }
            }

#pragma omp for reduction(+ : bari_num, bari_denom)
            for (int i = 0; i < NUM_FISH; i++)
            {
                // Update fish weight
                school[i].weight = fmin(W_MAX, school[i].weight + school[i].delta_fi / delta_fi_max);
                school[i].weight = fmax(0, school[i].weight);

                bari_num += calculateObjective(&school[i]) * school[i].weight;
                bari_denom += calculateObjective(&school[i]);
            }
        }
        double bari = bari_num / bari_denom;

        if (step % (NUM_STEPS / 20) == 0)
            printf("Step %d - Barycenter: %lf\n", step + 1, bari);
    }

    // Calculate and print elapsed time
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Elapsed Time: %lf seconds\n", elapsed_time);

    // Free dynamically allocated memory
    free(school);

    return 0;
}

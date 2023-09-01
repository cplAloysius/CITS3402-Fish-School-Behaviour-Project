#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_FISH 1000000 // Number of fish in the school
#define NUM_STEPS 100    // Number of simulation steps
#define W_INITIAL 10.0   // Initial weight of each fish
#define W_MAX (2 * W_INITIAL)

// Structure to represent a fish
typedef struct
{
    double x, y; // Coordinates
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
    clock_t start_time = clock();

    // Simulation loop
    for (int step = 0; step < NUM_STEPS; step++)
    {
        double delta_fi_max = 0.0;
        for (int i = 0; i < NUM_FISH; i++)
        {
            // Swim in a random direction
            // school[i].x += (double)(rand() % 21 - 10) / 100.0;
            // school[i].y += (double)(rand() % 21 - 10) / 100.0;

            // Calculate change in objective function
            double old_obj = calculateObjective(&school[i]);
            double new_x = school[i].x + (double)(rand() % 21 - 10) / 100.0;
            double new_y = school[i].y + (double)(rand() % 21 - 10) / 100.0;
            school[i].x = new_x;
            school[i].y = new_y;
            double new_obj = calculateObjective(&school[i]);
            double delta_fi = fabs(new_obj - old_obj);

            if (delta_fi > delta_fi_max)
            {
                delta_fi_max = delta_fi;
            }

            // Update fish weight
            school[i].weight = fmin(W_MAX, school[i].weight + delta_fi_max);
        }

        double bari_num = 0.0, bari_denom = 0.0, bari = 0.0;
        for (int i = 0; i < NUM_FISH; i++)
        {
            bari_num += sqrt((school[i].x * school[i].x) + (school[i].y * school[i].y)) * school[i].weight;
            bari_denom += sqrt((school[i].x * school[i].x) + (school[i].y * school[i].y));
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
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %lf seconds\n", elapsed_time);

    // Free dynamically allocated memory
    free(school);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_FISH 1000000 // Number of fish in the school
#define W_INITIAL 10.0   // Initial weight of each fish


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
    // Free dynamically allocated memory
    free(school);

    return 0;
}

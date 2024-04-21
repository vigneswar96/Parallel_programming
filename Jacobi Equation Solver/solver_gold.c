#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h"

/* Function definitions */
extern grid_t *copy_grid(grid_t *);

/* This function solves the Jacobi method on the CPU using a single thread. */
int compute_gold_jacobi(grid_t *grid)
{
    int num_iter = 0;
	int done = 0;
    int i, j;
	double diff;
	float old, new; 
    float eps = 1e-6; /* Convergence criteria. */
    int num_elements; 

    /* Set up ping-pong buffers. */
    grid_t *src_grid = grid;
    grid_t *dest_grid = copy_grid(grid);
    grid_t *tmp;
    grid_t *to_delete = dest_grid;
        
    while(!done) { /* While we have not converged yet. */
        diff = 0.0;
        num_elements = 0;

        for (i = 1; i < (src_grid->dim - 1); i++) {
            for (j = 1; j < (src_grid->dim - 1); j++) {
                old = src_grid->element[i * src_grid->dim + j]; /* Store old value of grid point. */
                /* Apply the update rule. */	
                new = 0.25 * (src_grid->element[(i - 1) * src_grid->dim + j] +\
                              src_grid->element[(i + 1) * src_grid->dim + j] +\
                              src_grid->element[i * src_grid->dim + (j + 1)] +\
                              src_grid->element[i * src_grid->dim + (j - 1)]);

                dest_grid->element[i * dest_grid->dim + j] = new; /* Update the grid-point value. */
                diff = diff + fabs(new - old); /* Calculate the difference in values. */
                num_elements++;
            }
        }
		
        /* End of an iteration. Check for convergence. */
        diff = diff/num_elements;
#ifdef DEBUG
        fprintf(stderr, "Jacobi iteration %d. DIFF: %f.\n", num_iter, diff);
#endif 
        num_iter++;
			  
        if (diff < eps) 
            done = 1;

        /* Flip the source and destination buffers. */
        tmp = src_grid;
        src_grid = dest_grid;
        dest_grid = tmp;
	}

    free((void *) to_delete->element);
    free((void *) to_delete);    
    return num_iter;
}


/* This function solves the Gauss-Seidel method on the CPU using a single thread. */
int compute_gold_gauss(grid_t *grid)
{
    int num_iter = 0;
	int done = 0;
    int i, j;
	double diff;
	float old, new; 
    float eps = 1e-6; /* Convergence criteria. */
    int num_elements; 
	
	while(!done) { /* While we have not converged yet. */
        diff = 0.0;
        num_elements = 0;

        for (i = 1; i < (grid->dim - 1); i++) {
            for (j = 1; j < (grid->dim - 1); j++) {
                old = grid->element[i * grid->dim + j]; /* Store old value of grid point. */
                /* Apply the update rule. */	
                new = 0.25 * (grid->element[(i - 1) * grid->dim + j] +\
                              grid->element[(i + 1) * grid->dim + j] +\
                              grid->element[i * grid->dim + (j + 1)] +\
                              grid->element[i * grid->dim + (j - 1)]);

                grid->element[i * grid->dim + j] = new; /* Update the grid-point value. */
                diff = diff + fabs(new - old); /* Calculate the difference in values. */
                num_elements++;
            }
        }
		
        /* End of an iteration. Check for convergence. */
        diff = diff/num_elements;
#ifdef DEBUG
        fprintf(stderr, "Gauss iteration %d. DIFF: %f.\n", num_iter, diff);
#endif
        num_iter++;
			  
        if (diff < eps) 
            done = 1;
	}
	
    return num_iter;
}


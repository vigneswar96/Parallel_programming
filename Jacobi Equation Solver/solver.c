
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#include "grid.h" 

extern int compute_gold_gauss(grid_t *);
extern int compute_gold_jacobi(grid_t *);
int compute_using_pthreads_jacobi(grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid(int, float, float);
grid_t *copy_grid(grid_t *);
void print_grid(grid_t *);
void print_stats(grid_t *);
double grid_mse(grid_t *, grid_t *);
void *parellel_jacobi(void *args);

pthread_barrier_t barrier;
pthread_mutex_t mutex_for_sum; /* Location of lock variable protecting sum */



typedef struct thread_data_s {
    int tid;            /* Thread identifier */
    int num_threads;    /* Number of threads in the worker pool */
	  int max_iter;
    int offset;         /* Starting offset for each thread within the vectors */ 
    int chunk_size;     /* Size of data to be processed by thread */
    grid_t *grid_1;       /* The grid_1 grid. */   
    grid_t *grid_2;       /* The grid_2 grid. */
    grid_t *grid_3;       /* The tmp grid, x */
    double *diff;
    int *num_iter;
} thread_data_t;



int main(int argc, char **argv)
{	
	if (argc < 5) {
        fprintf(stderr, "Usage: %s grid-dimension num-threads min-temp max-temp\n", argv[0]);
        fprintf(stderr, "grid-dimension: The dimension of the grid\n");
        fprintf(stderr, "num-threads: Number of threads\n"); 
        fprintf(stderr, "min-temp, max-temp: Heat applied to the north side of the plate is uniformly distributed between min-temp and max-temp\n");
        exit(EXIT_FAILURE);
    }
    
    /* Parse command-line arguments. */
    int dim = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    float min_temp = atof(argv[3]);
    float max_temp = atof(argv[4]);
    struct timeval start, stop;	

    
    /* Generate grids and populate them with initial conditions. */
 	grid_t *grid_1 = create_grid(dim, min_temp, max_temp);
    /* Grids 2 and 3 should have the same initial conditions as Grid 1. */
    grid_t *grid_2 = copy_grid(grid_1);
    grid_t *grid_3 = copy_grid(grid_1);

	/* Compute reference solutions using the single-threaded versions. */
    int num_iter;

	fprintf(stderr, "\nUsing the single threaded version of Gauss to solve the grid\n");
    gettimeofday(&start, NULL);
	num_iter = compute_gold_gauss(grid_1);
    gettimeofday(&stop, NULL);
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	fprintf(stderr, "Printing statistics for the interior grid points\n");
    print_stats(grid_1);
	
	fprintf(stderr, "\nUsing the single threaded version of Jacobi to solve the grid\n");
    gettimeofday(&start, NULL);
	num_iter = compute_gold_jacobi(grid_2);
    gettimeofday(&stop, NULL);
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	fprintf(stderr, "Printing statistics for the interior grid points\n");
    print_stats(grid_2);

	/* Use pthreads to solve the equation using the jacobi method. */
	fprintf(stderr, "\nUsing pthreads to solve the grid using the jacobi method\n");
	gettimeofday(&start, NULL);
    num_iter = compute_using_pthreads_jacobi(grid_3, num_threads);
    gettimeofday(&stop, NULL);
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);			
    fprintf(stderr, "Printing statistics for the interior grid points\n");
	print_stats (grid_3);
    
    // /* Compute grid differences. */
    // fprintf(stderr, "MSE between the single-threaded Gauss and Jacobi grids: %f\n", grid_mse(grid_1, grid_2));
    // fprintf(stderr, "MSE between the single-threaded Jacobi and multi-threaded Jacobi grids: %f\n", grid_mse(grid_2, grid_3));

	/* Free up the grid data structures. */
	free((void *) grid_1->element);	
	free((void *) grid_1); 
	free((void *) grid_2->element);	
	free((void *) grid_2);
    free((void *) grid_3);
    free((void *) grid_3->element);

	exit(EXIT_SUCCESS);
}


void *parellel_jacobi(void *args)
{
    int num_iter = 0;
	int done = 0;
    int i, j;
    int start;
    int stop;
    int num_elements = 0;
    double pdiff = 0.0;
    float old, new; 
    float eps = 1e-6;                                        /* Convergence criteria. */
    thread_data_t *thread_data = (thread_data_t *)args;      /* Typecast argument to pointer to thread_data_t structure */
	
    grid_t *grid_1 = thread_data->grid_1;
    grid_t *grid_2 = thread_data->grid_2;
    grid_t *grid_3 = thread_data->grid_3;


    while(!done) {                                           /* While we have not converged yet. */

        if ( thread_data-> tid == 0){                        /* This condition is only true when thread_id is 0; */
            *(thread_data->diff) = 0.0;
            *(thread_data->num_iter) += 1;                   /* This increments the values pointed to by 1 */
        }

        pthread_barrier_wait(&barrier);


        num_elements = 0;

        if ( thread_data->tid == 0 ){                       
            start = 1;
            stop = (thread_data->offset + thread_data->chunk_size); /* Add the current offset('thread_data -> offset') and the chunk size ('thread_data -> chunk_size') */

        } else if (thread_data->tid < (thread_data->num_threads - 1)) {
            start = thread_data->offset ;                           /* The start variable is set to the current offset value */
            stop = (thread_data->offset + thread_data->chunk_size) ;/* The stop variable is set to the sum of the current offset and the chunk size. */
 
        } else {
            start = thread_data->offset ;                             /* The start variable is set to the current offset value */
            stop =  (grid_1->dim - 1) ;                               /*  The stop variable is set to one less than the dimension of grid_1. */ 
        }
        

        for (i = start  ; i <  stop  ; i++) {
                for (j = 1; j < (grid_1->dim - 1); j++) {
                    old = grid_1->element[i * grid_1->dim + j]; /* Store old value of grid point. */
                    /* Apply the update rule. */	
                    new = 0.25 * (grid_1->element[(i - 1) * grid_1->dim + j] +\
                                grid_1->element[(i + 1) * grid_1->dim + j] +\
                                grid_1->element[i * grid_1->dim + (j + 1)] +\
                                grid_1->element[i * grid_1->dim + (j - 1)]);

                    grid_2->element[i * grid_2->dim + j] = new; /* Update the grid-point value. */
                    pdiff = pdiff + fabs(new - old); /* Calculate the difference in values. */
                    num_elements++;
                }
            }
        pdiff = pdiff/num_elements;
        /* Accumulate partial sums  */ 
        pthread_mutex_lock(&mutex_for_sum);
        *(thread_data->diff) = *(thread_data->diff) +  pdiff;
        pthread_mutex_unlock(&mutex_for_sum);

        pthread_barrier_wait(&barrier);
        
        
        if (*(thread_data->diff) < eps) {
            done = 1;
        }
            
        pthread_barrier_wait(&barrier);

        /* Flip the source and destination buffers. */
        grid_3 = grid_1;
        grid_1 = grid_2;
        grid_2 = grid_3;

    }
    return 0;
}
/* FIXME: Edit this function to use the jacobi method of solving the equation. The final result should be placed in the grid data structure. */
int compute_using_pthreads_jacobi(grid_t *grid, int num_threads)
{		
    int num_iter = 0;
    int i;
	  double diff = 0.0;
    /* Set up ping-pong buffers. */
    grid_t *grid_1 = grid;
    grid_t *grid_2 = copy_grid(grid);
    grid_t *grid_3;
    grid_t *to_delete = grid_2;

    /* Thread setup */
    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));    /* Data structure to store thread IDs */
    pthread_attr_t attributes;                                                      /* Thread attributes */
    pthread_attr_init(&attributes);     
    pthread_mutex_t mutex_for_sum;                                                  /* Lock for the shared variable sum */
    pthread_mutex_init(&mutex_for_sum, NULL); 

    /* Allocate heap memory for required data structures and create the worker threads */
    thread_data_t *thread_data = (thread_data_t *)malloc (sizeof(thread_data_t) * num_threads);
    pthread_barrier_init(&barrier,NULL,num_threads);


    
    int chunk_size = (int)floor((float)grid_1->dim/(float)num_threads);            /* Compute the chunk size */


    /* Fork point: create worker threads */
    for (i = 0; i < num_threads; i++) 
	{
        thread_data[i].tid = i;
        thread_data[i].offset = i * chunk_size;           /* Offset */
        thread_data[i].num_threads = num_threads;
        thread_data[i].chunk_size = chunk_size;          /* Introducing the chunk size. */
        thread_data[i].grid_1 = grid_1;
        thread_data[i].grid_2 = grid_2;
        thread_data[i].grid_3 = grid_3;
        thread_data[i].diff = &diff;
        thread_data[i].num_iter = &num_iter;

		
        if ((pthread_create(&thread_id[i],  &attributes, parellel_jacobi, (void *)&thread_data[i])) != 0) {
            perror("pthread_create");
            return -1;
        }
    }



    /* Join point: wait for worker threads to finish */	  
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);
		 
    /* Free dynamically allocated data structures */
    free((void *)thread_data);
    free((void *) to_delete->element);
    free((void *) to_delete);    
    return num_iter;
}


/* Create a grid with the specified initial conditions. */
grid_t* create_grid(int dim, float min, float max)
{
    grid_t *grid = (grid_t *)malloc (sizeof(grid_t));
    if (grid == NULL)
        return NULL;

    grid->dim = dim;
	fprintf(stderr, "Creating a grid of dimension %d x %d\n", grid->dim, grid->dim);
	grid->element = (float *) malloc(sizeof(float) * grid->dim * grid->dim);
    if (grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < grid->dim; i++) {
		for (j = 0; j < grid->dim; j++) {
            grid->element[i * grid->dim + j] = 0.0; 			
		}
    }

    /* Initialize the north side, that is row 0, with temperature values. */ 
    srand((unsigned)time(NULL));
	float val;		
    for (j = 1; j < (grid->dim - 1); j++) {
        val =  min + (max - min) * rand ()/(float)RAND_MAX;
        grid->element[j] = val; 	
    }

    return grid;
}

/* Creates a new grid and copies over the contents of an existing grid into it. */
grid_t* copy_grid(grid_t *grid) 
{
    grid_t *new_grid = (grid_t *)malloc(sizeof(grid_t));
    if (new_grid == NULL)
        return NULL;

    new_grid->dim = grid->dim;
	new_grid->element = (float *)malloc(sizeof(float) * new_grid->dim * new_grid->dim);
    if (new_grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < new_grid->dim; i++) {
		for (j = 0; j < new_grid->dim; j++) {
            new_grid->element[i * new_grid->dim + j] = grid->element[i * new_grid->dim + j] ; 			
		}
    }

    return new_grid;
}

/* Print grid to screen. */
void print_grid(grid_t *grid)
{
    int i, j;
    for (i = 0; i < grid->dim; i++) {
        for (j = 0; j < grid->dim; j++) {
            printf("%f\t", grid->element[i * grid->dim + j]);
        }
        printf("\n");
    }
    printf("\n");
}


/* Print out statistics for the converged values of the interior grid points, including min, max, and average. */
void print_stats(grid_t *grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0;
    int num_elem = 0;
    int i, j;

    for (i = 1; i < (grid->dim - 1); i++) {
        for (j = 1; j < (grid->dim - 1); j++) {
            sum += grid->element[i * grid->dim + j];

            if (grid->element[i * grid->dim + j] > max) 
                max = grid->element[i * grid->dim + j];

             if(grid->element[i * grid->dim + j] < min) 
                min = grid->element[i * grid->dim + j];
             
             num_elem++;
        }
    }
                    
    printf("AVG: %f\n", sum/num_elem);
	printf("MIN: %f\n", min);
	printf("MAX: %f\n", max);
	printf("\n");
}

/* Calculate the mean squared error between elements of two grids. */
double grid_mse(grid_t *grid_1, grid_t *grid_2)
{
    double mse = 0.0;
    int num_elem = grid_1->dim * grid_1->dim;
    int i;

    for (i = 0; i < num_elem; i++) 
        mse += (grid_1->element[i] - grid_2->element[i]) * (grid_1->element[i] - grid_2->element[i]);
                   
    return mse/num_elem; 
}



		


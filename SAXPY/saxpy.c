
#define _REENTRANT /* Make sure library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void *compute_gold_using_chunking_method(void *arg);
void compute_using_pthreads_v2(float *, float *, float, int, int);
void *compute_gold_using_striding_method(void *arg);
int check_results(float *, float *, int, float);

/* Input data for threaded functions */
typedef struct thread_chunking_data_input {
    int thread_id;            // Assigned Thread ID
    int num_threads, num_elements;    // Total number of threads (k), Total length of the array (n)
    float *x, *y, a;
    int size_of_chunk, chunk_offset;
} thread_chunking_data_input;

typedef struct thread_striding_data_input {
    int thread_id;
    int num_threads, num_elements;
    float *x, *y, a;
} thread_striding_data_input;

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: Number of elements in the input vectors\n");
        fprintf(stderr, "num-threads: Number of threads\n");
		exit(EXIT_FAILURE);
	}
	
    int num_elements = atoi(argv[1]); 
    int num_threads = atoi(argv[2]);

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    fprintf(stderr, "Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i]; 
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    fprintf(stderr, "\nCalculating SAXPY using reference solution\n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    compute_gold(x, y1, a, num_elements); 
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 1\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 2\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check results for correctness */
    fprintf(stderr, "\nChecking results for correctness\n");
    float eps = 1e-12;                                      /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");
 
    if (check_results(y1, y3, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");

	/* Free memory */ 
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);

    exit(EXIT_SUCCESS);
}

/* Compute reference soution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i]; 
}

/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads)
{
    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t)); // Array containing all thread IDs
    thread_chunking_data_input *thread_data = (thread_chunking_data_input *)malloc(num_threads * sizeof(thread_chunking_data_input)); // An array containing every argument for a thread

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements;
        thread_data[i].x = x;
        thread_data[i].y = y;
        thread_data[i].a = a;
        thread_data[i].size_of_chunk = (int)floor((float)num_elements/(float)num_threads);
        thread_data[i].chunk_offset = i * thread_data[i].size_of_chunk;
    }

    // Creation of Threads
    // The operation will alter y
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&thread_id[i], NULL, compute_gold_using_chunking_method, (void *)&thread_data[i]);
    }

    // now below is the code to join the thread
    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_id[i], NULL);
    }

    free((void *)thread_id);
    free((void *)thread_data);
}

void *compute_gold_using_chunking_method(void *arg) {
    thread_chunking_data_input *thread_data = (thread_chunking_data_input *)arg;    // to get all the input arguments

    int thread_id = thread_data->thread_id;
    int num_threads = thread_data->num_threads;
    int num_elements = thread_data->num_elements;
    float *x = thread_data->x;
    float *y = thread_data->y;
    float a = thread_data->a;
    int size_of_chunk = thread_data->size_of_chunk;
    int chunk_offset = thread_data->chunk_offset;

    int i;
    if (thread_id < num_threads - 1) { //any number of thread
        for (i = chunk_offset; i < chunk_offset + size_of_chunk; i++) {
            y[i] = a * x[i] + y[i];
        }
    } else { // final thread
        for (i = chunk_offset; i < num_elements; i++) {
            y[i] = a * x[i] + y[i];
        }
    }

    pthread_exit(NULL); // pthread_exit must be used to exit all the threads
}

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads)
{
    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t)); // Array containing all thread IDs

    thread_striding_data_input *thread_data = (thread_striding_data_input *)malloc(num_threads * sizeof(thread_striding_data_input)); // Array containing arguments for all threads

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements;
        thread_data[i].x = x;
        thread_data[i].y = y;
        thread_data[i].a = a;
    }

    // below is the code for creation of thread
    // below is the operation that will change the value of y
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&thread_id[i], NULL, compute_gold_using_striding_method, (void *)&thread_data[i]);
    }

    // to join the thread
    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_id[i], NULL);
    }

    free((void *)thread_id);
    free((void *)thread_data);
}

void *compute_gold_using_striding_method(void *arg) {
    thread_striding_data_input *thread_data = (thread_striding_data_input *)arg;    // to get the inputs for every arguments

    int thread_id = thread_data->thread_id;
    int num_threads = thread_data->num_threads;
    int num_elements = thread_data->num_elements;
    float *x = thread_data->x;
    float *y = thread_data->y;
    float a = thread_data->a;

    int i;
    for (i = thread_id; i < num_elements; i += num_threads) {
        y[i] = a * x[i] + y[i];
    }

    pthread_exit(NULL); // pthread_exit is used to end the pthread
}


/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }
    
    return 0;
}




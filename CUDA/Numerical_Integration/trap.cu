#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/* Include the kernel code */
#include "trap_kernel.cu"

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);

int main(int argc, char **argv) 
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s a b n\n", argv[0]);
        fprintf(stderr, "a: Start limit. \n");
        fprintf(stderr, "b: End limit\n");
        fprintf(stderr, "n: Number of trapezoids\n");
        exit(EXIT_FAILURE);
    }

    float a = atof(argv[1]); /* Left limit */
    float b = atof(argv[2]); /* Right limit */
    int n = atoi(argv[3]); /* Number of trapezoids */

    float h = (b-a)/(float)n; // Height of each trapezoid  
    printf("Number of trapezoids = %d\n", n);
    printf("Height of each trapezoid = %f\n", h);

    struct timeval start, stop;	
    gettimeofday(&start, NULL);
    double reference = compute_gold(a, b, n, h);
    gettimeofday(&stop, NULL);
    printf("Reference solution computed on the CPU = %f\n", reference);
    printf("CPU Execution time = %fs\n", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1e6);

    gettimeofday(&start, NULL);
    double gpu_result = compute_on_device(a, b, n, h);
    gettimeofday(&stop, NULL);
    printf("Solution computed on the GPU = %f\n", gpu_result);
    printf("GPU Execution time = %fs\n", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1e6);
} 

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, int n, float h)
{
    double *d_global_sum = NULL; // Pointer for the global sum on the device
    double h_global_sum = 0.0; // Host global sum
    size_t size = sizeof(double); // Size needed for the global sum

    // Allocate memory for the global sum on the device
    cudaMalloc((void **)&d_global_sum, size);
    cudaMemset(d_global_sum, 0, size); // Initialize the global sum to 0

    // Set up the execution configuration
    int threadsPerBlock = 256; 
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Ensure enough blocks to cover all trapezoids

      // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record event on the default stream (0) before kernel launch
    cudaEventRecord(start, 0);
    
    // Launch the Kernel 
    trap_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, n, h, d_global_sum);
    cudaDeviceSynchronize(); // Ensure completion before stopping timer
        // Record event on the default stream (0) after kernel has finished
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait until the event is completed

    // Calculate the elapsed time between events
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Check for any errors launching the kernel 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    } 

    // Copy the result back to host
    cudaMemcpy(&h_global_sum, d_global_sum, size, cudaMemcpyDeviceToHost);
  
    // Free device memory
    cudaFree(d_global_sum);

    return h_global_sum;
}

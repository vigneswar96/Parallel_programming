#ifndef _TRAP_KERNEL_H_
#define _TRAP_KERNEL_H_

#define THREAD_BLOCK_SIZE 1024          /* Size of a thread block */
#define NUM_BLOCKS 40                   /* Number of thread blocks */

/* Device function which implements the function. Device functions can be called from within other __device__ or __global__ functions, but cannot be called from the host. */ 
__device__ float fd(float x) 
{
     return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/* Kernel function */
__global__ void trap_kernel(float a, float b, int n, float h, double* global_sum) {
    __shared__ double sum_per_thread[THREAD_BLOCK_SIZE];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global index of the thread.
    double local_sum = 0.0; // Initialize the local sum for each thread.

    // Initialize all entries of sum_per_thread to 0 to prevent reading uninitialized values.
    sum_per_thread[threadIdx.x] = 0.0;
    __syncthreads(); // Ensure all threads have written their initial value before proceeding.

    if (index < n) {
        float x_i = a + index * h;
        float f_x_i = fd(x_i); // Calculate the function value at x_i.

        // Handle the first and last points by adding half of their values to local_sum.
        if (index == 0 || index == n-1) {
            local_sum += 0.5 * f_x_i;
        } else {
            local_sum += f_x_i;
        }
    }
    sum_per_thread[threadIdx.x] = local_sum; // Store the local sum into shared memory.
    __syncthreads(); // Wait for all threads to finish their part.

    // Perform parallel reduction in shared memory.
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
        }
        __syncthreads(); // Ensure all additions at one step are completed before moving to the next.
    }

    // Only the first thread in each block writes its result to global memory.
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, sum_per_thread[0] * h);
    }
}

#endif /* _TRAP_KERNEL_H_ */

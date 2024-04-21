This is an implementation of a Jacobi and Gauss-Seidel method for solving the Poisson equation on a 2D grid. The Poisson equation is a partial differential equation that describes the relationship between a function and its second-order partial derivatives. It has a wide range of applications in physics, engineering, and mathematics.
The Jacobi method, named after Carl Gustav Jacob Jacobi, is an iterative algorithm to solve a
system of linear equations. This technique has many applications in numerical computing. For
example, it can be used to solve the following differential equation in two variables x and y:

$$ \frac{\partial^2 f}{\partial x^2} +  \frac{\partial^2 f}{\partial y^2} = 0 $$


The above equation solves for the steady-state value of a function f(x, y) defined over a physical
two-dimensional (2D) space where f is a given physical quantity.
In this assignment, f represents
the heat as measured over a metal plate shown in Figure, and we wish to solve the following
problem: given that we know the temperature at the edges of the plate, what ends up being the
steady-state temperature distribution inside the plate?  
The code consists of two main parts:
* solver.c:
   * The compute_gold_jacobi() function implements the Jacobi method to solve the Poisson equation on a single thread.
   * The compute_gold_gauss() function implements the Gauss-Seidel method to solve the Poisson equation on a single thread.
   * The compute_using_pthreads_jacobi() function uses multiple threads (via pthreads) to solve the Poisson equation using the Jacobi method.
* grid.h and grid.c:
   * The grid.h file defines the grid_t data structure, which represents the 2D grid.
   * The grid.c file contains functions for creating, copying, and printing the grid, as well as computing the mean squared error (MSE) between two grids. 
The main program in solver.c demonstrates the usage of the Jacobi and Gauss-Seidel solvers. It creates a 2D grid, populates it with initial conditions, and then solves the Poisson equation using the single-threaded and multi-threaded versions. The execution time, number of iterations, and statistics of the converged solutions are printed to the console. \

- **Parallelization:** The main challenge is to efficiently parallelize the Jacobi method using pthreads. This requires careful consideration of the data dependencies, thread synchronization, and load balancing to achieve good performance.

- **Convergence criteria:** Uses a simple absolute difference-based convergence criterion
- **Boundary conditions:** The current implementation assumes Dirichlet boundary conditions (fixed temperatures) on the north side of the grid. Handling more complex boundary conditions, such as Neumann or mixed conditions, would be a useful extension.
- **Numerical stability and accuracy:** Depending on the problem parameters (grid size, temperature range, etc.), the numerical stability and accuracy of the Jacobi and Gauss-Seidel methods may be a concern. Investigating alternative numerical schemes, such as the Conjugate Gradient method, could improve the robustness of the solver.

**Potential accomplishments** 

- **Efficient parallel implementation:** Developing an optimized parallel Jacobi solver that achieves good scalability and performance on modern multi-core systems would be a significant accomplishment. This involves techniques like task-level parallelism, data partitioning, and effective thread synchronization.
-**Generalization to other PDE problems:** The core concepts and techniques used in this Poisson equation solver could be extended to solve other types of partial differential equations, such as the heat equation or the wave equation. Demonstrating the flexibility and adaptability of the solver would be an impressive accomplishment.

 The below part explains the code implementation in detail: 
 
'''pthread_barrier_t barrier;

pthread_mutex_t mutex_for_sum; /* Location of lock variable protecting sum */
'''
These are the global varaiables that will be used for synchronization between the worker threads. 
'''
'''
pthread_barrier_t barrier; ''' is a type that represents a barrier, which is a synchronization point where threads must wait for each other to reach before they can continue. 

```
 pthread_mutex_t
```
pthread_mutex_t is a type that represents a mutex, which is a mutual exclusion lock used to protect shared resources. 
#######################################################################################

```
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
```


This defines a structure called thread_data_t that will be used to pass data to the worker threads. It includes information such as the thread ID, the number of threads, the maximum number of iterations, the starting offset for each thread, the chunk size of data to be processed by each thread, pointers to the three grids, a pointer to the difference value, and a pointer to the number of iterations. 

```
grid_t *grid_1 = create_grid(dim, min_temp, max_temp);
grid_t *grid_2 = copy_grid(grid_1);
grid_t *grid_3 = copy_grid(grid_1);
```

Three grid structures (grid_1, grid_2 and grid_3) are created. grid_1 is created using the `create_grid` function, which initializes the grid with the given dimensions and temperature range. grid_2 and grid_3 are created by copying the contents of grid_1 using the copy_grid function. 

```
int num_iter;
fprintf(stderr, "\nUsing the single threaded version of Gauss to solve the grid\n");
gettimeofday(&start, NULL);
num_iter = compute_gold_gauss(grid_1);
gettimeofday(&stop, NULL);
printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
            + (stop.tv_usec - start.tv_usec)/(float)1000000));
fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);
fprintf(stderr, "Printing statistics for the interior grid points\n");
print_stats(grid_1);
```

This block solves the grid using the single-threaded Gauss method. The `compute_gold_gauss` function is called to solve the grid, and the number of iterations(num_iter) required for convergence is stored. The execution time is measured using the `gettimeofday` function, and the statistics for the interior grid points are printed using the `print_stats` function. 


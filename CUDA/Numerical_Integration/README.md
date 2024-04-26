***Numerical Integration using Trapezoidal Rule on GPU***

This project implements the trapezoidal rule for numerical integration on a GPU using CUDA. The goal is to estimate the area under a curve defined by a function f(x) over the interval [a, b] by dividing the interval into n subintervals and approximating the area using trapezoids.

Given a function f(x) and end points a and b, where a < b, we wish to estimate the area under this
curve; that is, we wish to determine R &int;<sub>a</sub><sup>b</sup> f(x) dx.

The trapezoidal rule approximates the integral as follows:
∫<sub>a</sub><sup>b</sup> f(x) dx ≈ h[f(x<sub>0</sub>)/2 + f(x<sub>1</sub>) + f(x<sub>2</sub>) + ... + f(x<sub>n-1</sub>) + f(x<sub>n</sub>)/2]
where h = (b - a) / n is the length of each subinterval, and x<sub>i</sub> = a + ih for i = 0, 1, ..., n.


***Implementation***
The project consists of a serial CPU implementation (trap_gold.cpp) and a parallel GPU implementation (trap.cu). The GPU implementation utilizes CUDA to parallelize the computation of the integral using the following approach:

1) The execution grid is divided into ***k 1D thread blocks***.
2) Each thread in the execution grid strides over the trapezoids and calculates a `partial sum`, storing it in shared memory.
3) ***Barrier synchronization*** ensures that shared memory is populated with the partial sums.
4) Each thread block reduces the values in shared memory to a single value using a ***tree-style reduction*** technique.
5) A designated thread within each thread block accumulates the reduced value into a shared variable in GPU global memory using `atomic operations`.

***Results***

The GPU implementation achieves significant speedup over the serial CPU version. The speedup is reported for different numbers of trapezoids (10<sup>4</sup>, 10<sup>6</sup>, and 10<sup>8</sup>) used to estimate the integral of the function f(x) = (√(1 + x<sup>2</sup>)) / (1 + x<sup>4</sup>) over the interval [5, 100]. 

***Usage***

To build and run the project, follow these steps:

Compile the CPU implementation: `g++ -o trap_gold trap_gold.cpp`
Compile the GPU implementation: `nvcc -o trap trap.cu`
Run the CPU implementation: `./trap_gold a b n`
Run the GPU implementation: `./trap a b n`

Replace a, b, and n with the desired lower limit, upper limit, and number of trapezoids, respectively.

***Report***

The project report discusses the design of the GPU kernel, optimization techniques used, speedup achieved over the serial version, and the sensitivity of the kernel to thread-block size in terms of execution time.

###############################################################################################################

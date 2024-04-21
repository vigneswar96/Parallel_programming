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
steady-state temperature distribution inside the plate? \   
<img src="Screenshot 2024-04-21 163108.jpg" width="250" height="150" />

The 2D space of interest is first discretized via a uniform grid in which $$\Delta$$ is the spacing—for example,
in millimeters—between grid points along the two Cartesian dimensions. If $$\Delta$$ is sufficiently
small, we can approximate the second-order derivatives in the euqation shown above using the Taylor series as 

<img src="Screenshot 2024-04-21 183222.jpg" width="400" height="150" />

Substituting the above equations into the first equation we obtain 

<img src="Screenshot 2024-04-21 163108.jpg" width="250" height="150" />





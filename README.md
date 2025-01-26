Conjugate Gradient Method Solver
================================

A simple CUDA program that solves the matrix equation:
`A.x = b`, for `x`. Where `x` and `b` are real-valued column vectors of size `N` and `A` is a sparse `NxN` matrix with three single-valued diagonals of the form:
```
╭                        ╮
│ -2  1  0  0    0  0  0 │
│  1 -2  1  0 …  0  0  0 │
│  0  1 -2  1    0  0  0 │
│  0  0  1 -2    0  0  0 │
│     ⋮        ⋱    ⋮    │
│  0  0  0  0   -2  1  0 │
│  0  0  0  0 …  1 -2  1 │
│  0  0  0  0    0  1 -2 │
╰                        ╯
```

Building
========

Prerequisites for building on Linux/Windows:
- CUDA toolkit 11+
- CMake 3.11+

Clone the repo and open a shell in the new directory:
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
This should generate a `conjgrad` executable in the `build` directory.

Testing
=======

The `test/` directory contains three test inputs for the `b` vector that can be used to verify the algorithm. As well as the expected resulting `x` vector in the corresponding `*.expected.txt` files.

The program contains both a simple, serial CPU-only implementation that is selected using the `-s` argument, as well as the parallelized CUDA implementation, selected using the `-p` argument.

All test data and expected results have been generated using `numpy`.

CUES
====

CUDA Epidemic Simulator is a CUDA parallelized implementation of an algorithm to simulate epidemic spreading in large networks.

Author: Luka Šterbić, luka.sterbic@fer.hr

1) Dependencies
---------------------

To properly run CUES the following software is needed:

1. bash shell
2. nvcc 6.5+ (and compatible gcc or clang version)
3. make
    
A CUDA capable GPU with computing capability 3.0 (Kepler architecture) or greater is also needed.

This software was developed and tested on a GeForce GT 650M with 2 multiprocessors and 192 cores per MP.
The warp size for the Kepler architecture is of 32 threads and the maximum number of threads per MP is 2048.


2) Installation
---------------------

To instal CUES run make in it's root folder. This will build an executable named `cues`.


3) Usage
---------------------

`./cues <path_to_graph> <source_vertex> <simulations>`

CUES expects 3 command line arguments. The first one is the path to the graph of the network that will be used in the simulation.
The second parameter is the vertex ID of the start of the epidemics (patient zero), while the third parameter controls the number of simulations to perform.

4) Acknowledgments
---------------------

The BFS parallelization used in CUES is largely based on paper by Duane Merrill, Michael Garland and Andrew Grimshaw,
 [Scalable GPU Graph Traversal][1]


[1]: https://research.nvidia.com/publication/scalable-gpu-graph-traversal "Scalable GPU Graph Traversal"
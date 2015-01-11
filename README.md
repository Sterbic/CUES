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
The program will automatically choose the best compatible device in terms of GFLOPS.

This software was developed and tested on a GeForce GT 650M with 2 multiprocessors and 192 cores per MP.
The warp size for the Kepler architecture is of 32 threads and the maximum number of threads per MP is 2048.


2) Installation
---------------------

To install CUES run make in its root folder. This will build an executable named `cues`.


3) Usage
---------------------

`./cues <graph_path> <source_node> <p> <q> <simulations> <out_dir>`

CUES expects 5 command line arguments: 
- **graph_path:** the path to the graph in edge list format of the network that will be used
in the simulation 
- **source_node:** the node ID of the start of the epidemics, patient zero 
- **p:** the probability that a node will infect its neighbors 
- **q:** the probability that a node will recover and become immune 
- **simulations:** the number of simulations to run
- **out_dir:** the directory where the output will be saved

For each simulation, CUES will output the iteration in which each node was infected.

To test the program after installation run the following command from the root folder: `./test.sh`


4) Documentation
---------------------

A more detailed explanation of the usage and implemented algorithm may be viewed [here][2].


5) Acknowledgments
---------------------

The BFS parallelization used in CUES is largely based on a paper by Duane Merrill, Michael Garland and Andrew Grimshaw: [Scalable GPU Graph Traversal][1].


[1]: https://research.nvidia.com/publication/scalable-gpu-graph-traversal "Scalable GPU Graph Traversal"
[2]: https://docs.google.com/document/d/1m2SnumwScQOHD21op-_IcjQMCoVw34bIVtU_sIQ1VcE/edit?usp=sharing "CUES Documentation"

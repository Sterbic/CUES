/**
 * Graph structure and functions for the CUES project.
 */

#ifndef GRAPH_CUH_
#define GRAPH_CUH_

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "utils.cuh"

using namespace std;

/**
 * The network graph used in the simulation. Stored in CSR format.
 * N: number of nodes
 * M: number of edges
 * R[N+1]: contains the indices in C for the start of the adjacency lists
 * CSize: the size of the concatenated adjacency lists, 2 * M
 * C[CSize]: concatenated adjacency lists
 */
typedef struct {
	unsigned int N;
	unsigned int M;
	unsigned int *R;
	unsigned int CSize;
	unsigned int *C;
} Graph;

/**
 * Load the a graph stored in edge list format.
 * path: the path to the input file
 * returns: a pointer to the created graph
 */
Graph *loadGraph(const char *path) {
	FILE *file = fopen(path, "r");
	exitIf(file == NULL, "The input file could not be opened.");

	char line[100];
	unsigned int first, second;
	unsigned int maxIndex = 0;
	unsigned int M = 0;

	vector<vector <unsigned int> > adjacencyLists(0);

	while(fgets(line, 100, file) != NULL) {
		if(line[0] == '#') {
			continue;
		}

		if(sscanf(line, "%u %u", &first, &second) == 2) {
			maxIndex = max(maxIndex, max(first, second));
			M++;

			while(adjacencyLists.size() < maxIndex + 1) {
				adjacencyLists.push_back(vector<unsigned int>(0));
			}

			adjacencyLists[first].push_back(second);
			adjacencyLists[second].push_back(first);
		}
	}

	fclose(file);

	Graph *graph = (Graph *) malloc(sizeof(Graph));
	exitIf(graph == NULL, "Error allocating graph structure.");

	graph->N = maxIndex + 1;
	graph->M = M;
	graph->R = (unsigned int *) malloc((graph->N + 1) * sizeof(unsigned int));
	graph->CSize = 2 * M;
	graph->C = (unsigned int *) malloc(graph->CSize * sizeof(unsigned int));

	exitIf(graph->R == NULL || graph->C == NULL,
			"Error allocating R and C arrays.");

	int cIndex = 0;

	for(int i = 0; i < adjacencyLists.size(); i++) {
		graph->R[i] = cIndex;

		for(int j = 0; j < adjacencyLists[i].size(); j++) {
			graph->C[cIndex] = adjacencyLists[i][j];
			cIndex++;
		}
	}

	graph->R[graph->N] = cIndex;
	return graph;
}

/**
 * Free the momory allocated by a call to loadGraph.
 * graph: the graph returned by load graph
 */
void freeGraph(Graph *graph) {
	if(graph != NULL) {
		if(graph->R != NULL) {
			free(graph->R);
		}

		if(graph->C != NULL) {
			free(graph->C);
		}

		free(graph);
	}
}

/**
 * Prints the R and C arrays.
 * grpah: the graph to print
 */
void printGraph(Graph *graph) {
	printf("N: %u\nM: %u\nR: ", graph->N, graph->M);

	for(int i = 0; i < graph->N + 1; i++) {
		printf("%u", graph->R[i]);
		printf(i == graph->N ? "\n" : ", ");
	}

	printf("C: ");

	for(int i = 0; i < graph->CSize; i++) {
		printf("%u", graph->C[i]);
		printf(i == graph->CSize - 1 ? "\n" : ", ");
	}
}

#endif

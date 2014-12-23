"""
Graph compression script.

Utility script for the CUES project to compress the given graph, either
directed or undirected, to the best format for CUES to use it. The
script will extract the connected component of the graph accessible
from the given source node. In the output graph the node representing
the source in the original graph will have ID 0.

Usage:
    python3 compress_graph.py input source output

Args:
    input: the path to the input graph file
    source: the source node from which the epidemic will start
    output: the path to the output graph file
"""
__author__ = 'Luka Sterbic'

import sys
from collections import deque
from os.path import basename


class Graph(object):

    class Node(object):

        def __init__(self, node_id):
            self.node_id = node_id
            self.compressed_id = None
            self.neighbors = []

    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.compressed_nodes = {}
        self.compressed_id = 0

    def _get_or_create_node(self, node_id):
        if node_id in self.nodes:
            return self.nodes[node_id]
        else:
            node = Graph.Node(node_id)
            self.nodes[node_id] = node
            return node

    def _add_edge(self, first_id, second_id):
        first_node = self._get_or_create_node(first_id)
        second_node = self._get_or_create_node(second_id)

        first_node.neighbors.append(second_id)
        second_node.neighbors.append(first_id)

    def _get_next_compressed_id(self):
        compressed_id = self.compressed_id
        self.compressed_id += 1
        return compressed_id

    def compress(self, source):
        if source not in self.nodes:
            raise ValueError("The source ID is not in the graph.")

        queue = deque([self.nodes[source]])

        while len(queue) > 0:
            node = queue.popleft()

            if node.compressed_id is not None:
                continue

            node.compressed_id = self._get_next_compressed_id()
            self.compressed_nodes[node.compressed_id] = node

            Graph._print_progress(self.compressed_id, "nodes compressed")

            for neighbor_id in node.neighbors:
                queue.append(self.nodes[neighbor_id])

    def dump_to_file(self, path):
        with open(path, "w") as file:
            source_id = self.compressed_nodes[0].node_id

            file.write("# CUES compressed graph\n")
            file.write("# Original graph: %s\n" % self.name)
            file.write("# Source node in original graph: %d\n\n" % source_id)

            for compressed_id in range(self.compressed_id):
                Graph._print_progress(compressed_id + 1, "nodes saved")
                node = self.compressed_nodes[compressed_id]

                for neighbor_id in node.neighbors:
                    neighbor = self.nodes[neighbor_id]

                    if compressed_id < neighbor.compressed_id:
                        file.write("%d %d\n" % (compressed_id,
                                                neighbor.compressed_id))

    @staticmethod
    def _print_progress(state, description):
        print("\r%d %s" % (state, description), end="")

    @staticmethod
    def load_from_file(path):
        with open(path, "r") as file:
            graph = Graph(basename(path))
            counter = 0

            for line in file:
                counter += 1
                Graph._print_progress(counter, "lines processed")

                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) != 2:
                    continue

                first = int(parts[0])
                second = int(parts[1])

                graph._add_edge(first, second)

            return graph


def main(input_path, source, output_pah):
    print("Reading graph...")
    graph = Graph.load_from_file(input_path)

    print("\nCompressing graph...")
    graph.compress(int(source))

    print("\nSaving graph to file...")
    graph.dump_to_file(output_pah)

    print("\nGraph successfully compressed!")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        exit(1)

    main(*sys.argv[1:])

"""
Graph BFS script.

Utility script for the CUES project to check the correctness of the
CUES output. The script parses the given graph and performs a BFS from
the specified source node. The output is a file with the level of each
node in a single line.

Usage:
    python3 bfs_graph.py input source output

Args:
    input: the path to the input graph file
    source: the BFS source node
    output: the path to the output file
"""
__author__ = 'Luka Sterbic'

import sys
from collections import deque


class Graph(object):

    class Node(object):

        def __init__(self, node_id):
            self.node_id = node_id
            self.level = None
            self.neighbors = []

        def __repr__(self):
            return str(self.node_id)

    def __init__(self):
        self.nodes = {}

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

    def bfs(self, source):
        queue = deque([self.nodes[source], None])
        level = 0
        nodes_visited = 0

        while len(queue) > 0:
            node = queue.popleft()

            if node is None:
                level += 1
                if len(queue) > 0:
                    queue.append(None)
            elif node.level is None:
                node.level = level
                nodes_visited += 1

                Graph._print_progress(nodes_visited, "nodes visited")

                for neighbor_id in node.neighbors:
                    neighbor = self.nodes[neighbor_id]

                    if neighbor.level is None:
                        queue.append(neighbor)

        Graph._print_progress(nodes_visited, "nodes visited", True)

    def dump_levels(self, path):
        with open(path, "w") as file:
            for node_id in range(len(self.nodes)):
                Graph._print_progress(node_id + 1, "levels saved")
                file.write("%d\n" % self.nodes[node_id].level)

            Graph._print_progress(len(self.nodes), "levels saved", True)

    @staticmethod
    def _print_progress(state, description, force=False):
        if force or state % 100 == 0:
            print("\r%d %s" % (state, description), end="")

    @staticmethod
    def load_from_file(path):
        with open(path, "r") as file:
            graph = Graph()
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

                if first < 0 or second < 0:
                    raise ValueError("Node IDs should be positive integers.")

                graph._add_edge(first, second)

            Graph._print_progress(counter, "lines processed", True)

            return graph


def main(input_path, source, output_pah):
    print("Reading graph...")
    graph = Graph.load_from_file(input_path)

    print("\nRunning BFS...")
    graph.bfs(int(source))

    print("\nSaving levels to file")
    graph.dump_levels(output_pah)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        exit(1)

    main(*sys.argv[1:])

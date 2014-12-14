#! /usr/bin/env python3

"""
Graph generating script.

Utility script for the CUES project to generate well known graphs such
as chains, cycles and complete graphs with the given number of nodes.

Usage:
    python3 make_graph.py type nodes path

Args:
    type: the type of the graph to create, [chain, cycle, complete]
    nodes: the total number of nodes in the graph
    path: the path to the output file where the graph will be saved
"""
__author__ = "Luka Sterbic"

import sys

GRAPH_TYPES = {"chain", "cycle", "complete"}


def main(graph_type, nodes, output):
    if graph_type not in GRAPH_TYPES:
        raise ValueError("Unknown graph type.")

    nodes = int(nodes)
    if nodes < 3:
        raise ValueError("The number of nodes must be at least 3.")

    with open(output, "w") as file:
        file.write("# %s_%d\n\n" % (graph_type, nodes))

        if graph_type == "chain":
            _create_chain(nodes, file)
        elif graph_type == "cycle":
            _create_cycle(nodes, file)
        elif graph_type == "complete":
            _create_complete(nodes, file)


def _create_chain(nodes, output_file):
    for node in range(nodes - 1):
        output_file.write("%d %d\n" % (node, node + 1))


def _create_cycle(nodes, output_file):
    _create_chain(nodes, output_file)
    output_file.write("%d %d\n" % (0, nodes - 1))


def _create_complete(nodes, output_file):
    for first_node in range(nodes):
        for second_node in range(first_node + 1, nodes):
            output_file.write("%d %d\n" % (first_node, second_node))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        exit(1)

    main(*sys.argv[1:])

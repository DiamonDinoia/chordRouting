import networkx as nx
import random
import argparse
from collections import OrderedDict
import hashlib
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import collections
import operator
import sys

nodes = None
filename = None
log_file = None
length = None


# Helper function to determine if a key falls within a range
def in_range(key, a, b):
    # is c in (a,b) mod (2**bits)
    return (a < key < b) if b > a else (key > a or key < b)


def binary_search(l, elem):
    first = 0
    last = len(l)
    mid = (last-first)/2
    while first != last:
        mid = (last-first)/2
        if l[mid] == elem:
            return mid
        elif l[mid] < elem:
            last = mid
        elif l[mid] > elem:
            first = mid
    while l[mid] > elem:
        mid -= 1
    return mid


class Node:
    bits = 0
    verbose = False
    stats = False

    @staticmethod
    def get_id(id, exp, n):
        if exp == 0:
            return (id+1) % n
        return (id + (2 << (exp-1))) % n

    def __init__(self, id):
        self.id = id
        self.n = n
        self.finger_table = OrderedDict()
        self.successor = None
        self.predecessor = None

    def init_finger_table(self, nodes):
        for i in range(0, self.bits):
            ind = Node.get_id(self.id, i, self.n)
            node = ind
            while node not in nodes:
                node = (node + 1) % self.n
            self.finger_table[ind] = nodes[node]
            if i == 0:
                self.successor = nodes[node]
        node = self.id
        while True:
            node = (node-1) % self.n
            if node in nodes:
                self.predecessor = nodes[node]
                break

    def get_neighbours(self):
        return self.finger_table.values()

    def close_preceding_node(self, id):
        neighbours = list(self.get_neighbours())
        neighbours.reverse()
        for node in neighbours:
            if in_range(node.id, self.id, id):
                return node
        return self

    def find_successor(self, id, nodes=None):
        if nodes is not None:
            nodes.append(self)
        if Node.verbose:
            print("node: " + str(self.id))
        if in_range(id, self.id, self.successor.id+1):
            return self
        node = self.close_preceding_node(id)
        return node.find_successor(id, nodes)

    def __repr__(self):
        t = 'Node ' + str(self.id) + ' Finger table\n'
        t += tabulate([(x, y.id) for x, y in self.finger_table.items()], headers=['Address', 'Node'], tablefmt='orgtbl')
        return str(t) + '\n'

    def __eq__(self, other):
        return self.id == other.id


class Coordinator:

    def init_nodes(self, nodes, bits, n):
        Node.bits = bits
        while len(self.nodes) < nodes:
            print(len(self.nodes), file=sys.stderr)
            # key = random.randrange(0, n)
            key = sha1(random.randrange(0, n))
            if key not in self.nodes:
                self.nodes[key] = Node(key)
                # print(key, file=sys.stderr)
            # else:
            #     print('Nooo', file=sys.stderr)

    def __init__(self, nodes, bits, n):
        self.nodes = OrderedDict()
        self.init_nodes(nodes, bits, n)
        for node in self.nodes.values():
            node.init_finger_table(self.nodes)

    def get_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(self.nodes.keys(), reverse=True))
        for node in self.nodes.values():
            graph.add_edges_from([(node.id, x.id) for x in node.get_neighbours()])
        return graph

    def write_graph(self, path):
        nx.write_gml(self.get_graph(), path)

    def print_ring(self):
        g = self.get_graph()
        nx.draw_circular(g, dim=2, with_labels=True, node_size=120, font_size=8)

    def __repr__(self):
        out = ' '.join(['Nodes:', str(nodes), 'Bits:', str(bits)])
        out += '\n'
        for node in self.nodes.values():
            out += str(node)
        return out + '\n'


def calculate_density(graph):
    return nx.density(graph)


# def number_connected_components(graph):
#     return nx.number_connected_components(graph.to_undirected())


def in_degree_histogram(graph, figure=None):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, color='b')

    plt.title("In Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    # ax.set_xticks([d + 0.4 for d in deg])
    # ax.set_xticklabels(deg)

    if figure is None:
        plt.show()
    else:
        plt.savefig(fig)


def out_degree_histogram(graph, figure=None):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Out degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    if figure is None:
        plt.show()
    else:
        plt.savefig(fig)


def calculate_eccentricity(graph):
    return sum(nx.eccentricity(graph).values())/len(nx.eccentricity(graph).values())


def calculate_diameter(graph):
    return nx.diameter(graph)


def queries_histogram(queries, figure=None):
    nodes = [item.id for sublist in queries for item in sublist[1:]]
    nodes = collections.Counter(nodes)
    nodes = [x for x in nodes.values()]
    nodes.sort(reverse=True)
    fig, ax = plt.subplots()
    plt.bar(range(len(nodes)), nodes, width=0.80, color='b')

    plt.title("Routing Histogram")
    plt.ylabel("Number of queries routed")
    ax.set_xticklabels([])
    if figure is None:
        plt.show()
    else:
        plt.savefig(fig)


def last_hop_histogram(queries, figure=None):
    nodes = [item[-1].id for item in queries]
    nodes = collections.Counter(nodes)
    nodes = sorted(nodes.items(), key=operator.itemgetter(1), reverse=True)
    node, cnt = zip(*nodes)
    fig, ax = plt.subplots()
    plt.bar(range(len(cnt)), cnt, color='b')
    plt.title("Last Hop Histogram")
    plt.ylabel("Number of queries routed")
    if figure is None:
        plt.show()
    else:
        plt.savefig(fig)


def main():
    coordinator = Coordinator(nodes, bits, n)
    print('Initialized finished', file=sys.stderr)
    graph = coordinator.get_graph()
    print('Graph generated', file=sys.stderr)
    nx.write_gml(graph, filename)
    print('Graph saved', file=sys.stderr)
    # print(number_connected_components(graph))
    # print(coordinator)
    # coordinator.print_ring()
    # plt.show()
    in_degree_histogram(graph, 'in_degree_histogram')
    out_degree_histogram(graph, 'out_degree_histogram')
    print('Simulation started', file=sys.stderr)
    hops = []
    for _ in range(length):
        key = sha1(random.randrange(0, n))
        node = np.random.choice(list(coordinator.nodes.keys()))
        hops.append([])
        coordinator.nodes[node].find_successor(key, hops[-1])
        with open(log_file, 'w') as f:
            print(*[x.id for x in hops[-1]], file=f)
    print('Simulation finished', file=sys.stderr)
    queries_histogram(hops, 'queries_histogram')
    last_hop_histogram(hops, 'last_hop_histogram')
    hops = [len(x)-1 for x in hops]
    print('Average path length', sum(hops)/len(hops))
    print('Density', calculate_density(graph))
    print('Eccentricity', calculate_eccentricity(graph))
    print('Diameter', calculate_diameter(graph))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', type=int, default=1000)
    parser.add_argument('-b', '--bits', type=int, default=11)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-f', '--filename', type=str, default='./network')
    parser.add_argument('-l', '--log_file', type=str, default='./logfile.txt')
    parser.add_argument('-d', '--length', type=int, default='10000')
    args = parser.parse_args()
    bits = args.bits
    n = 2 << (bits-1)
    nodes = args.nodes
    filename = args.filename
    random.seed(args.seed)
    log_file = args.log_file
    length = args.length
    print('Nodes', nodes)
    print('Bits', bits)

    def sha1(key, n=n):
        return int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % n
        # return key

    main()

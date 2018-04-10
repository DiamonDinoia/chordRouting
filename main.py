import networkx as nx
import random
import argparse
from collections import OrderedDict
import hashlib
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import collections

bits = None
nodes = None
filename = None
log_file = None


# Helper function to determine if a key falls within a range
def in_range(key, a, b):
    # is c in (a,b) mod (2**bits)
    return (a < key < b) if b > a else (key > a or key < b)


class Node:
    bits = 0
    verbose = False
    stats = False

    @staticmethod
    def get_id(id, exp, bits):
        return (id + (2 ** exp)) % (2 ** bits)

    def __init__(self, id):
        self.id = id
        self.finger_table = OrderedDict()
        self.successor = None
        self.predecessor = None

    def init_finger_table(self, nodes):
        for i in range(0, bits):
            ind = Node.get_id(self.id, i, self.bits)
            node = ind
            while node not in nodes:
                node = (node + 1) % (2 ** bits)
                assert(0 <= node < (2 ** bits))
            self.finger_table[ind] = nodes[node]
            if i == 0:
                self.successor = nodes[node]
        node = self.id
        while True:
            node = (node-1) % (2**bits)
            assert(node >= 0)
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

    def init_nodes(self, nodes, bits):
        Node.bits = bits
        while len(self.nodes) < nodes:
            key = sha1(random.randrange(0, 2 ** bits))
            assert(0 <= key < 2 ** bits)
            if key not in self.nodes:
                self.nodes[key] = Node(key)

    def __init__(self, nodes, bits):
        self.nodes = OrderedDict()
        self.init_nodes(nodes, bits)
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


def number_connected_components(graph):
    return nx.number_connected_components(graph.to_undirected())


def in_deegree_histogram(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("In Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()


def out_deegree_histogram(graph):
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

    plt.show()


def calculate_eccentricity(graph):
   return sum(nx.eccentricity(graph).values())/len(nx.eccentricity(graph).values())


def calculate_diameter(graph):
    return nx.diameter(graph)


def main():
    coordinator = Coordinator(nodes, bits)
    graph = coordinator.get_graph()
    print(calculate_density(graph))
    print(number_connected_components(graph))
    in_deegree_histogram(graph)
    out_deegree_histogram(graph)
    print(calculate_eccentricity(graph))
    print(calculate_diameter(graph))
    # print(coordinator)
    # coordinator.print_ring()
    # plt.show()
    hops = []
    for _ in range(10000):
        key = sha1(random.randrange(0, 2**bits))
        node = np.random.choice(list(coordinator.nodes.keys()))
        hops.append([])
        coordinator.nodes[node].find_successor(key, hops[-1])
        # print(*[x.id for x in hops[-1]])
    hops = [len(x)-1 for x in hops]
    # print(*hops)
    print(sum(hops)/len(hops))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', type=int, default=1000)
    parser.add_argument('-b', '--bits', type=int, default=11)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-f', '--filename', type=str, default='./network')
    parser.add_argument('-l', '--log_file', type=str, default='./logfile.txt')
    args = parser.parse_args()
    print(args)
    bits = args.bits
    nodes = args.nodes
    filename = args.filename
    random.seed(args.seed)

    def sha1(key, bits=bits):
        return int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % (2 ** bits)

    main()

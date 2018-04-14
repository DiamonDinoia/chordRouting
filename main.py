import networkx as nx
import random
import argparse
from collections import OrderedDict
import hashlib
from tabulate import tabulate
import matplotlib.pyplot as plt
import collections
import operator
import sys

nodes = None
filename = None
log_file = None
length = None
plot = None


# Helper function to determine if a key falls within a range
def in_range(key, a, b):
    # is c in (a,b) mod (2**bits)
    return (a < key < b) if b > a else (key > a or key < b)


# Helper function used to determine the closest preceeding node
# when initializing the finger tables
def binary_search(l, elem):
    first = 0
    last = len(l)-1
    while first <= last:
        mid = (first + last) / 2
        mid = int(mid)
        if l[mid] >= elem:
            last = mid-1
        else:
            first = mid+1
    # print(l[mid])
    if first >= len(l):
        return l[0]
    return l[first]


# Class that models the node in the network
class Node:
    bits = 0
    verbose = False
    stats = False
    keys = None

    # Static method used to determine the address of the finger table
    @staticmethod
    def get_id(id, exp, n):
        if exp == 0:
            return (id+1) % n
        return (id + (2 << (exp-1))) % n

    def __init__(self, id):
        self.id = id
        self.n = n
        self.finger_table = OrderedDict()
        self.neighbours = None
        self.successor = None
        self.predecessor = None

    # Since the network is assumed static after the creation of all the nodes this function is called in order to
    # fill the nodes finger table
    def init_finger_table(self, nodes):
        for i in range(0, self.bits):
            ind = Node.get_id(self.id, i, self.n)
            # Optimization in order to find the correct node
            node = binary_search(Node.keys, ind)
            self.finger_table[ind] = nodes[node]

        node = binary_search(Node.keys, self.id+1 % self.n)
        self.successor = nodes[node]
        node = binary_search(Node.keys, self.id-1 % self.n)
        self.predecessor = nodes[node]
        self.neighbours = list(self.finger_table.values())
        self.neighbours.reverse()

    # return the neighbours of a node
    def get_neighbours(self):
        return self.finger_table.values()

    # Implemented fallowing the chord pseudocode, it returns the closest preceding node of id
    def close_preceding_node(self, id):
        for node in self.neighbours:
            if in_range(node.id, self.id, id):
                return node
        return self

    # implemented fallowing the chord pseudocode, it finds the successor od id
    # Nodes is a parameters used to keep track of the route taken by the query
    def find_successor(self, id, nodes=None):
        if nodes is not None:
            nodes.append(self)
        if Node.verbose:
            print("node: " + str(self.id))
        if in_range(id, self.id, self.successor.id+1):
            return self
        node = self.close_preceding_node(id)
        return node.find_successor(id, nodes)

    # Override of the to string method in order to have a nice representation of the node
    def __repr__(self):
        t = 'Node ' + str(self.id) + ' Finger table\n'
        t += tabulate([(x, y.id) for x, y in self.finger_table.items()], headers=['Address', 'Node'], tablefmt='orgtbl')
        return str(t) + '\n'

    # Override of the equal method
    def __eq__(self, other):
        return self.id == other.id


# Utility class used to initialize and perform the simulation
class Coordinator:

    # Helpler function that create and initialize all the nodes of the swarm
    def init_nodes(self, nodes, bits, n):
        Node.bits = bits
        while len(self.nodes) < nodes:
            key = sha1(random.randrange(0, n))
            if key not in self.nodes:
                self.nodes[key] = Node(key)
        tmp = self.nodes
        self.nodes = OrderedDict()
        ordered_keys = sorted(tmp.keys())
        for key in ordered_keys:
            self.nodes[key] = tmp[key]

    # Simple constructor that perform basic initialization of the data structures
    def __init__(self, nodes, bits, n):
        self.nodes = OrderedDict()
        self.init_nodes(nodes, bits, n)
        Node.keys = list(self.nodes.keys())
        for node in self.nodes.values():
            # print(node.id, file=sys.stderr)
            node.init_finger_table(self.nodes)

    # returns a networkx graph representing the chord network
    def get_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(self.nodes.keys(), reverse=True))
        for node in self.nodes.values():
            graph.add_edges_from([(node.id, x.id) for x in node.get_neighbours()])
        return graph

    # export the chord network in gml format in order to analize it with exernal programs like cytoscape
    def write_graph(self, path):
        nx.write_gml(self.get_graph(), path)

    # plots a graphical representation of the network
    def print_ring(self):
        g = self.get_graph()
        nx.draw_circular(g, dim=2, with_labels=True, node_size=120, font_size=8)

    # override the toString method
    def __repr__(self):
        out = ' '.join(['Nodes:', str(nodes), 'Bits:', str(bits)])
        out += '\n'
        for node in self.nodes.values():
            out += str(node)
        return out + '\n'

# Utility funtion that calculates the density of the netwotk
def calculate_density(graph):
    return nx.density(graph)


# Utility funciton that plots an histogram of the in degree distribution
def in_degree_histogram(graph, figure=None):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, color='b')

    plt.title("In Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")

    if figure is None:
        plt.show()
    else:
        plt.savefig(figure)


# Utility funciton that plots an histogram of the out degree distribution
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
        plt.savefig(figure)


# Utility function that calculates the eccentricity of the network
def calculate_eccentricity(graph):
    return sum(nx.eccentricity(graph).values())/len(nx.eccentricity(graph).values())


# Utility function that calculates the diameter of the network
def calculate_diameter(graph):
    return nx.diameter(graph)


# Utility funciton that plots an histogram of the queries histogram
# basically the number of queries routed by each node
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
        plt.savefig(figure)


# Utility funciton that plots an histogram of the queries handled by each node
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
        plt.savefig(figure)


def main():
    coordinator = Coordinator(nodes, bits, n)
    print('Initialization finished', file=sys.stderr)
    graph = coordinator.get_graph()
    print('Graph generated', file=sys.stderr)
    nx.write_gml(graph, filename)
    print('Graph saved', file=sys.stderr)
    if plot:
        coordinator.print_ring()
        plt.show()
    in_degree_histogram(graph, './in_degree_histogram')
    out_degree_histogram(graph, './out_degree_histogram')
    print('Simulation started', file=sys.stderr)
    hops = []
    keys = list(coordinator.nodes.keys())
    with open(log_file, 'w') as f:
        for _ in range(length):
            key = sha1(random.randrange(0, n))
            node = random.randrange(0, nodes)
            hops.append([])
            coordinator.nodes[keys[node]].find_successor(key, hops[-1])
            print(*[x.id for x in hops[-1]], file=f)
    print('Simulation finished', file=sys.stderr)
    queries_histogram(hops, './queries_histogram')
    last_hop_histogram(hops, './last_hop_histogram')
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
    parser.add_argument('-p', '--plot', type=bool, default=False)
    args = parser.parse_args()
    bits = args.bits
    n = 2 << (bits-1)
    nodes = args.nodes
    filename = args.filename
    random.seed(args.seed)
    log_file = args.log_file
    length = args.length
    plot = args.plot
    print('Nodes', nodes)
    print('Bits', bits)

    #Helper function to complute the sha1
    def sha1(key, n=n):
        return int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % n

    main()

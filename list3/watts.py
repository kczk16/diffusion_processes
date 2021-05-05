import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from scipy.stats import binom
import networkx as nx
import math
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from random import random, randint
from scipy.special import comb

n = 2000
p = 0.5
k = 4
beta = 0.3

def initial(n, k):
    '''
    function that generates graph with initial conditions for Watts-Strogatz algorithm
    :param n: number of edges
    :type n: int
    :param k: number of initial connections for each node
    :type k: int
    :return: list of nodes, list of edges
    '''
    nodes = [i for i in range(n)]
    edges = []
    for node in nodes:
        for i in range(1, int(k/2)+1):
            if node + i >= len(nodes):
                index1 = node - i
                index2 = abs(len(nodes) - node - i)
                destination_nodes = [nodes[index1], nodes[index2]]
            else:
                index1 = node - i
                index2 = node + i
                destination_nodes = [nodes[index1], nodes[index2]]
            edges.append([node, destination_nodes[0]])
            edges.append([node, destination_nodes[1]])
    return nodes, edges


def edges_per_node(nodes, edges):
    '''
    function that transforms list of edges into a dictionary with nodes as keys and list of nodes that they are
    connected to as values
    :param nodes: list of nodes
    :type nodes: list
    :param edges: list of edges
    :type edges: list
    :rtype: dict
    '''
    edges_per_node = {node : [] for node in nodes}
    for edge in edges:
        for node in edge:
            index = edge.index(node)
            if index and edge[0] not in edges_per_node[node]:
                edges_per_node[node].append(edge[0])
            if not index and edge[1] not in edges_per_node[node]:
                edges_per_node[node].append(edge[1])

    return edges_per_node



def my_custom_random(start,stop,exclude):
    '''
    function that generates random number until it is different from a number we want to  exclude
    :param start: interval start
    :type start: int
    :param stop: interval stop
    :type stop: int
    :param exclude: value we want to exclude
    :type exclude: int
    :return: random number different from exclude param value
    :rtype: int
    '''
    randInt = randint(start,stop)
    return my_custom_random(start,stop,exclude) if randInt in exclude else randInt


def moving_edge(edges_per_node, n, beta, k):
    '''
    function modifying connections in graph
    :param edges_per_node: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    :param n: number of nodes
    :type n: int
    :param beta: probability
    :type beta: float
    :param k: number of connections per node
    :type k: int
    :return: modified dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    '''
    not_initial_edges = []
    for node in edges_per_node:
         s = len(edges_per_node[node])
         for a in range(s):
             if [node,edges_per_node[node][a]] in not_initial_edges or [edges_per_node[node][a],node] in not_initial_edges:
                 continue
             else:
                x = random()
                if x < beta:
                    toVert = my_custom_random(0, n-1,[node])

                    if toVert in edges_per_node[node]:
                        continue

                    else:
                        A = edges_per_node[node][a]
                        edges_per_node[node].append(toVert)
                        edges_per_node[toVert].append(node)
                        edges_per_node[node].remove(edges_per_node[node][a])

                        edges_per_node[A].remove(node)
                        not_initial_edges.append([node, edges_per_node[node][a]])

    degrees = {i: 0 for i in edges_per_node.keys()}
    for edge in edges_per_node.values():
        for vert in edge:
            degrees[vert] += 1
    plt.hist(degrees.values(), bins=range(min(degrees.values()),max(degrees.values())+1), density=True, rwidth=0.8)
    plt.xlabel('Number of degrees')
    plt.ylabel('Frequency')
    plt.title('Watts Graph')

    p_k = []
    for j in range(min(degrees.values()), max(degrees.values()) + 1):
        f = min(j - k / 2, k / 2)
        p = 0
        for i in range(int(f + 1)):
            p += comb(int(k / 2), i) * (1 - beta) ** i * beta ** (int(k / 2) - i) * (
                        ((beta * int(k / 2)) ** (j - i - int(k / 2))) / math.factorial(j - i - int(k / 2))) * math.exp(
                -beta * int(k / 2))
        p_k.append(p)


    plt.plot(np.array(range(min(degrees.values()), max(degrees.values()) + 1)) + 0.5, p_k, 'ro')
    plt.show()
    return edges_per_node


def make_edgelist(edgedict):
    '''
    function that transforms edges dictionary to list of tuples
    :param edgedict: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    :return: list of tuples
    '''
    edgelist = []
    for node in edgedict.keys():
        for i in range(len(edgedict[node])):
            if (node, edgedict[node][i]) in edgelist or (edgedict[node][i], node) in edgelist:
                continue
            else:
                edgelist.append((node, edgedict[node][i]))

    return edgelist


def plot_graph(graph):
    '''
    function that plots graph
    :param graph: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    :type graph: dict
    '''
    G = nx.Graph()
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    li = make_edgelist(graph)
    ax.set_title('watts')
    G.add_edges_from(li, with_labels=True)
    nx.draw(G)
    plt.show()

nodes, initial_edges = initial(n, k)
edges_per_node_val = edges_per_node(nodes, initial_edges)
moving_edge(edges_per_node_val, n, beta, k)

n=20
nodes, initial_edges = initial(n, k)
edges_per_node_val = edges_per_node(nodes, initial_edges)
plot_graph(moving_edge(edges_per_node_val, 20, beta, k))
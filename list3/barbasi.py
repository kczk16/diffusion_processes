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


def initial_barbasi(n):
    '''
    function that generates graph with initial conditions for Barabasi-Albert algorithm
    :param n: number of nodes
    :type n: int
    :return: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    '''
    edges_per_node = {node : [] for node in range(n)}
    number_of_edges = n - 1
    for node in edges_per_node.keys():
        for end in range(number_of_edges):
            if node != end and end not in edges_per_node[node]:
                edges_per_node[node].append(end)

    return  edges_per_node

# zad 1 c)
def add_new_node(given_node, edges_per_node, k):
    '''
    function that adds new node to graph according to probability depending on nodes degrees
    :param given_node: node to be added
    :param edges_per_node: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    :param k: number of connections for new node
    :return: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    '''
    prob_per_node = {}
    degrees_sum = sum([len(v) for v in edges_per_node.values()])
    for node in edges_per_node:
        prob_per_node[node] = len(edges_per_node[node])/degrees_sum
    z = 0
    while z < k:
        x = random()
        prob_range = [0, prob_per_node[0]]
        for node in prob_per_node:
            if node > 0 :
                start = prob_range[1]
                stop = prob_range[1] + prob_per_node[node]
                prob_range = [start, stop]
            if x >= prob_range[0] and x < prob_range[1]:
                toVert = node
                if given_node in edges_per_node.keys():
                    if toVert in edges_per_node[given_node]:
                        continue
                    else:
                        edges_per_node[toVert].append(given_node)
                        edges_per_node[given_node].append(toVert)
                        z += 1
                else:
                    edges_per_node[given_node] = [toVert]
                    edges_per_node[toVert].append(given_node)
                    z += 1
    return edges_per_node



def Barbasi_Albert_graph(initial_number, n, k):
    '''
    function that generates graph accirding to Barbasi Albert method
    :param initial_number: number of initial nodes
    :type initial_number: int
    :param n: number of nodes to be added
    :type n: int
    :param k: number of connections per node
    :type k: int
    :return: dictionary with nodes as keys and and list of nodes that they are
    connected to as values
    '''
    edges_per_node = initial_barbasi(initial_number)
    for i in range(initial_number, n):
        edges_per_node = add_new_node(i, edges_per_node, k)

    degrees = {i: 0 for i in edges_per_node.keys()}
    for edge in edges_per_node.values():
        for vert in edge:
            degrees[vert] += 1
    count, bins_count = np.histogram([float(i) for i in degrees.values()] , bins=10)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    y = 1 - cdf

    plt.plot(bins_count[1:], y, label="1-CDF")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
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
    ax.set_title('Barbasi')
    G.add_edges_from(li, with_labels=True)
    nx.draw(G)
    plt.show()


barabasi = Barbasi_Albert_graph(4, 2000, 2)
plot_graph(Barbasi_Albert_graph(4, 20, 2))
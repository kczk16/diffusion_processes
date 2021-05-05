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


def random_graph(n, p=1):
    '''
    function that generates random graph
    :param n: number of nodes
    :type n: int
    :param p: probability
    :type p: float
    :return: list of nodes, list of edges
    '''
    nodes = []
    edgeList = []
    for j in range(n):
        if j in nodes:
            pass
        else:
            nodes.append(j)

    for vert in nodes:
        index = nodes.index(vert)
        for i in range(index + 1, n):
            x = random()
            if x < p:
                edgeList.append([vert, nodes[i]])

    degree_per_vert = {i: 0 for i in nodes}
    for edge in edgeList:
        for vert in edge:
            degree_per_vert[vert] += 1
    plt.hist(degree_per_vert.values(), density=True, rwidth=0.8)
    x = np.linspace(np.min(list(degree_per_vert.values())), np.max(list(degree_per_vert.values())),
                    np.max(list(degree_per_vert.values())) - np.min(list(degree_per_vert.values())) + 1)
    plt.plot(x, binom.pmf(x, n, p), 'r*', ms=5, label='binom pmf')
    plt.show()
    return nodes, edgeList


def degree_and_variance(nodes, edgeList):
    '''
    function that calculates variance and average degree of a node
    :param nodes: list of nodes
    :param edgeList: list of edges
    :return: variance, average degree of a node
    '''
    degree_per_vert = {i: 0 for i in nodes}
    for edge in edgeList:
        for vert in edge:
            degree_per_vert[vert] += 1
    avg_degree = np.mean(degree_per_vert.values())
    variance = np.variance(degree_per_vert.values())

    return variance, avg_degree


def frequency_hist(nodes, edgeList):
    '''
    function that generates frequency histogram
    :param nodes: list of nodes
    :param edgeList: list of edges
    '''
    edges_per_vert = {i: 0 for i in nodes}
    for edge in edgeList:
        for vert in edge:
            edges_per_vert[vert] += 1

    plt.bar(edges_per_vert.keys(), edges_per_vert.values())
    plt.title('histogram')
    plt.show()


def plot_graph(graph):
    '''
    function that plots graph
    :param graph: list of edges
    '''
    G = nx.Graph()
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    li = []
    for i in range(len(graph)):
        a = tuple(graph[i])
        li.append(a)
        ax.set_title('Random')

    G.add_edges_from(li, with_labels=True)
    nx.draw(G)
    plt.show()

random_graph(n, p)
nodes, edges = random_graph(20, p)

plot_graph(edges)
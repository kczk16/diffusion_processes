import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import subprocess
import os
from random import randint, random


def get_state(graph):
    """
    Gets state of nodes in the graph
    :param graph: networkx graph
    :return: lists of nodes per state: S, I, R
    """
    nodes_list = list(graph.nodes(data=True))
    S, I, R = [], [], []
    for node in range(len(nodes_list)):
        if graph.nodes[node]['state'] == 'S':
            S.append(node)
        if graph.nodes[node]['state'] == 'I':
            I.append(node)
        if graph.nodes[node]['state'] == 'R':
            R.append(node)
    return S, I, R


def SIR_network(graph, beta=0.2, start_node=None):
    """
    Simulates SIR on a network graph
    :param graph: networkx graph
    :param beta: probability of transmission
    :param start_node: first infected node
    :return: networkx graph, list of nodes with states S, I, R in time
    """
    nodes = list(graph.nodes(data=True))
    nx.set_node_attributes(graph, values="S", name="state")
    if start_node is not None:
        patient_zero = start_node
    else:
        patient_zero = randint(0, len(nodes) - 1)
    graph.nodes[patient_zero]["state"] = "I"
    S, I, R = get_state(graph)
    S_in_time, I_in_time, R_in_time = [S], [I], [R]
    while len(I) > 0:
        for inf_node in I:
            neighbors = graph.neighbors(inf_node)
            for neighbor in neighbors:
                if graph.nodes[neighbor]["state"] == "S" and random() < beta:
                    graph.nodes[neighbor]["state"] = "I"
            graph.nodes[inf_node]["state"] = "R"

        S, I, R = get_state(graph)
        S_in_time.append(S)
        I_in_time.append(I)
        R_in_time.append(R)

    return graph, S_in_time, I_in_time, R_in_time

#a)
#SIR_network(nx.barabasi_albert_graph(25, 4))
#SIR_network(nx.erdos_renyi_graph(50, 0.5))
#SIR_network(nx.watts_strogatz_graph(25, 2, 0.5))

graph_list = [nx.erdos_renyi_graph(50, 0.5), nx.barabasi_albert_graph(50, 4), nx.watts_strogatz_graph(50, 2, 0.5)]
name_list = ['random', 'barabasi', 'watts strogatz']

#b)
def monte_carlo(graph, p, N, start_node):
    """
    Simulates SIR model on network graph N times
    :param graph: networkx graph
    :param p: probability of transmission
    :param N: number of executions for monte carlo
    :param start_node: first infected node
    :return: lists of average nodes infected in time, average durations of epidemics
    """
    nodes = len(graph.nodes())
    infected = []
    time_of_epidemic = []
    fractions = []
    for n in range(N):
        graph, S_in_time, I_in_time, R_in_time = SIR_network(graph, p, start_node)
        infected.append(I_in_time)

    for epidemic in infected:
        f = []
        time_of_epidemic.append(len(epidemic))
        for time_step in epidemic:
            f.append(len(time_step)/nodes)
        fractions.append(f)

    max_time = max(time_of_epidemic)
    avg_time_of_epidemic = np.average(time_of_epidemic)
    avg_infected_in_time = [0]*max_time
    for ep_frac in fractions:
        for frac in range(len(ep_frac)):
            avg_infected_in_time[frac] += ep_frac[frac]

    avg_infected_in_time = [i/len(fractions) for i in avg_infected_in_time]
    return avg_infected_in_time, avg_time_of_epidemic

#print(monte_carlo(nx.erdos_renyi_graph(50, 0.5), 0.2, 10, 0))

def plot_avg(graph_list, p_list, N, start_node):
    """
    Plots average fraction of infected nodes
    :param graph: networkx graph
    :param p_list: list of probabilities of transmission
    :param N: number of executions for monte carlo
    :param start_node: first infected node
    """
    for graph in graph_list:
        for p in p_list:
            avg_infected_in_time, _ = monte_carlo(graph, p, N, start_node)
            plt.plot(avg_infected_in_time, label="p = {}, model = {}".format(p, graph_list.index(graph)))
    plt.title('Average fraction of infected nodes')
    plt.xlabel('time')
    plt.ylabel('fraction of infected nodes')
    plt.legend()
    plt.show()

#plot_avg(graph_list,[0.2, 0.5, 0.9], 100, 0)

#d)
def total_infected(graphs_list, p_number, N):
    """
    Plots average proportion of the network that becomes infected
    :param graphs_list: list of networkx graphs
    :param p_number: number of values of probabilities of transmission that will be generated in linspace
    :param N: number of executions for monte carlo
    """
    p_list = np.linspace(0, 1, p_number)
    for graph in graphs_list:
        avg_infected = []
        for p in p_list:
            all_infected = []
            for i in range(N):
                graph, S_in_time, I_in_time, R_in_time = SIR_network(graph, beta=p)
                for epidemic_state in I_in_time:
                    all_infected.append(len(epidemic_state))
            avg = sum(all_infected)/N
            avg_infected.append(avg)
        plt.plot(p_list, avg_infected, label=name_list[graph_list.index(graph)])
    plt.title("Average number of nodes of the network that become infected")
    plt.legend()
    plt.show()


#total_infected(graph_list, 20, 50)


def time_to_clear_infection(graphs_list, p_number, N):
    """
    Plots average time to clear infection
    :param graphs_list: list of networkx graphs
    :param p_number: number of values of probabilities of transmission that will be generated in linspace
    :param N: number of executions for monte carlo
    """
    for graph in graphs_list:
        nodes = len(graph.nodes())
        p_list = np.linspace(0, 1, p_number)
        avg_time = []
        for p in p_list:
            _, avg_time_of_epidemic = monte_carlo(graph, p, N, randint(0, nodes - 1))
            avg_time.append(avg_time_of_epidemic)
        plt.plot(p_list, avg_time, label=name_list[graph_list.index(graph)])
    plt.title("Average time to clear infection")
    plt.legend()
    plt.show()

#time_to_clear_infection(graph_list, 20, 50)

def time_to_most_infected(graphs_list, p_number, N):
    """
    Plots average time to most infected nodes
    :param graphs_list: list of networkx graphs
    :param p_number: number of values of probabilities of transmission that will be generated in linspace
    :param N: number of executions for monte carlo
    """
    p_list = np.linspace(0, 1, p_number)
    for graph in graphs_list:
        avg_time = []
        for p in p_list:
            time = []
            for i in range(N):
                graph, S_in_time, I_in_time, R_in_time = SIR_network(graph, beta=p)
                lengths = [len(epidemic_state) for epidemic_state in I_in_time]
                t = lengths.index(max(lengths))
                time.append(t)
            avg = sum(time) / N
            avg_time.append(avg)
        plt.plot(p_list, avg_time, label=name_list[graph_list.index(graph)])
    plt.title("Average time to most infected nodes")
    plt.legend()
    plt.show()

#time_to_most_infected(graph_list, 20, 50)

def SIR_network_plot(graph, beta=0.2, start_node=None):
    """
    Plots SIR on a network graph
    :param graph: networkx graph
    :param beta: probability of transmission
    :param start_node: first infected node
    :return: networkx graph, list of nodes with states S, I, R in time
    """
    nodes = list(graph.nodes(data=True))
    nx.set_node_attributes(graph, values="S", name="state")
    if start_node is not None:
        patient_zero = start_node
    else:
        patient_zero = randint(0, len(nodes) - 1)
    graph.nodes[patient_zero]["state"] = "I"
    S, I, R = get_state(graph)
    S_in_time, I_in_time, R_in_time = [S], [I], [R]
    
    counter = 0
    pos=nx.circular_layout(graph)
    nx.draw_networkx_edges(graph,pos)        
    nx.draw_networkx_nodes(graph,pos,nodelist=S_in_time[0],node_color='#c8c8c8')
    nx.draw_networkx_nodes(graph,pos,nodelist=I_in_time[0],node_color='r') 
    nx.draw_networkx_nodes(graph,pos,nodelist=R_in_time[0],node_color='g') 
    plt.axis("off")
    filename = str(counter) + ".png"
    plt.savefig(filename)
    plt.close()
    
    while len(I) > 0:
        counter = counter + 1
        for inf_node in I:
            neighbors = graph.neighbors(inf_node)
            for neighbor in neighbors:
                if graph.nodes[neighbor]["state"] == "S" and random() < beta:
                    graph.nodes[neighbor]["state"] = "I"
            graph.nodes[inf_node]["state"] = "R"

        S, I, R = get_state(graph)
        S_in_time.append(S)
        I_in_time.append(I)
        R_in_time.append(R)

        nx.draw_networkx_edges(graph,pos)        
        nx.draw_networkx_nodes(graph,pos,nodelist=S_in_time[counter],node_color='#c8c8c8')
        nx.draw_networkx_nodes(graph,pos,nodelist=I_in_time[counter],node_color='r') 
        nx.draw_networkx_nodes(graph,pos,nodelist=R_in_time[counter],node_color='g') 
        plt.axis("off")
        filename = str(counter) + ".png"
        plt.savefig(filename)
        plt.close()

#SIR_network_plot(nx.barabasi_albert_graph(30, 5))
#SIR_network_plot(nx.erdos_renyi_graph(30, 0.25))
#SIR_network_plot(nx.watts_strogatz_graph(30, 4, 0.7), beta = 0.5)


def grid2gif(image_str, output_gif):
    '''
    Function that generates gif file using imagemagic
    '''
    str1 = '/usr/local/bin/convert -delay 100 -loop 1 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)
    
#grid2gif("*.png", "my_output.gif")


def file_remove():
    """
    Function that removes png files
    """
    for file in os.listdir("/Users/kczk/Desktop/list5"):
        if file.endswith('.png'):
            os.remove(file)
#file_remove()
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

def _initialize_graph(graph_type, no_of_nodes):
    '''
    function that returns a networkx graph instance with specific type
    :param graph_type: type of graph
    :type graph_type: str
    :param no_of_nodes: number of nodes in graph
    :type no_of_nodes: int
    :return: networkx graph instance
    '''
    if 'random' in graph_type.lower():
        return nx.erdos_renyi_graph(no_of_nodes, 0.9)
    elif 'barabasi' in graph_type.lower():
        return nx.barabasi_albert_graph(no_of_nodes, 3)
    elif 'watts' in graph_type.lower():
        return nx.watts_strogatz_graph(no_of_nodes, 3, 0.9)

def random_walk(graph_type, no_of_steps, no_of_nodes, start_node):
    '''
    function that generates random walk on nodes of a graph
    :param graph_type: type of graph
    :type graph_type: str
    :param no_of_steps: number of steps in walk
    :type no_of_steps: int
    :param no_of_nodes: number of nodes in graph
    :type no_of_nodes: int
    :param start_node: node from which the random walk will be started
    :type start_node: int
    :return: networkx graph, steps in walk as a list of nodes
    '''
    graph = _initialize_graph(graph_type, no_of_nodes)
    walk = [start_node]
    while len(walk) < no_of_steps:
        next_step = random.choice([n for n in graph.neighbors(walk[-1])])
        walk.append(next_step)
    return graph, walk

def animate_graph(graph, walk):
    '''
    function that animates a random walk on a graph
    :param graph: networkx graph
    :param walk: steps in walk as a list of nodes
    :type walk: list
    :return: animation
    '''
    n = len(list(graph.nodes))
    colors = []
    for node in walk:
        color = [0 for i in range(n)]
        color[node] = 1
        colors.append(color)
    colors = np.array(colors)
    pos = nx.spring_layout(graph)
    labels = {i : i for i in list(graph.nodes)}
    nodes = nx.draw_networkx_nodes(graph, pos)
    edges = nx.draw_networkx_edges(graph, pos)
    def animate_frame(i):
        nodes.set_array(colors[i])
        nodes.set_cmap('RdYlBu')
        nx.draw_networkx_labels(graph, pos, labels)
        return nodes,
    fig = plt.gcf()
    plt.title('random walk')
    anim = FuncAnimation(fig, animate_frame, interval=550, frames=len(colors))
    plt.show()
    return anim

graph, walk = random_walk('barabasi', 15, 10, 0)
#graph, walk = random_walk('random', 15, 10, 0)
#graph, walk = random_walk('watts', 15, 10, 0)

#ani = animate_graph(graph, walk)
#ani.save('graph_watts.gif', writer='imagemagick', fps=6)

def _get_no_of_first_hits(graph, walk):
    '''
    function that gets number of first hits of nodes in a graph during a random walk
    :param graph: networkx graph
    :param walk: steps in walk as a list of nodes
    :type walk: list
    :return: list with numbers of first hits per node
    '''
    n = len(list(graph.nodes))
    hits = [0 for i in range(n)]
    for step in walk[1:]:
        if hits[step] != 0 :
            continue
        hits[step] += walk.index(step)
    return hits

print(walk)
print(_get_no_of_first_hits(graph, walk))

def avg_hitting_nodes(graph_type, no_of_nodes, start_node, no_of_steps, N):
    '''
    function that gets average number of hits of nodes in a graph during a random walk
    :param graph_type: type of graph
    :type graph_type: str
    :param no_of_nodes: number of nodes in a graph
    :type no_of_nodes: int
    :param start_node: node from which the random walk will be started
    :type start_node: int
    :param no_of_steps: number of steps in a random walk
    :type no_of_steps: int
    :param N: number of repetitions of a random walk
    :type N: int
    :return: list with average numbers of hits per node
    '''
    all_hits = []
    for i in range(N):
        graph, walk = random_walk(graph_type, no_of_steps, no_of_nodes, start_node)
        hits = _get_no_of_first_hits(graph, walk)
        all_hits.append(hits)
    avg_hits = [0 for i in range(no_of_nodes)]
    for i in range(no_of_nodes):
        for j in range(no_of_nodes):
            try:
                avg_hits[j] += all_hits[i][j]
            except IndexError:
                continue
    avg_hits = [i/N for i in avg_hits]
    return avg_hits

data = avg_hitting_nodes('barabasi', 10, 0, 10, 30)
print(data)

def plot_hist_with_dif_graph_types(no_of_nodes, start_node, no_of_steps):
    '''
    function that makes a barplot of average hits per node in a number of random walks per every graph type
    :param no_of_nodes: number of nodes in a graph
    :type no_of_nodes: int
    :param start_node: node from which the random walk will be started
    :type start_node: int
    :param no_of_steps: number of steps in a random walk
    :type no_of_steps: int
    '''
    barWidth = 0.9
    bars1 = avg_hitting_nodes('barabasi', no_of_nodes, start_node, no_of_steps, 200)
    bars2 = avg_hitting_nodes('watts', no_of_nodes, start_node, no_of_steps, 200)
    bars3 = avg_hitting_nodes('random', no_of_nodes, start_node, no_of_steps, 200)
    r1 = [i for i in range(1, no_of_nodes+1)]
    r2 = [i for i in range(no_of_nodes+1, no_of_nodes*2+1)]
    r3 = [i for i in range(no_of_nodes*2+1, no_of_nodes*3+1)]
    r4 = r1 + r2 + r3
    plt.bar(r1, bars1, width=barWidth, label='barabasi')
    plt.bar(r2, bars2, width=barWidth, label='watts')
    plt.bar(r3, bars3, width=barWidth, label='random')
    plt.legend()
    names = [str(i) for i in range(no_of_nodes)]
    names = names*3
    plt.xticks([r + barWidth for r in range(len(r4))], names, rotation=90)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.title('average hits per node in the random walk with starting node {}'.format(start_node))
    plt.show()

plot_hist_with_dif_graph_types(10, 0, 200)


def plot_of_avg_hits_of_specific_node(no_of_nodes, start_node, no_of_steps, node, N):
    '''
    function that makes a plot of distributions of avg. no. of hits of specific node in a random walk
    :param no_of_nodes: number of nodes in a graph
    :type no_of_nodes: int
    :param start_node: node from which the random walk will be started
    :type start_node: int
    :param no_of_steps: number of steps in a random walk
    :type no_of_steps: int
    :param node: node of which we want to make a plot of average number of hits
    :type node: int
    :param N: number of walks to be made
    :type N: int
    '''
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    dictionary = {1: 'barabasi', 2: 'watts', 3: 'random'}
    Bars1, Bars2, Bars3 = [], [], []
    for i in range(N):
        bars1 = avg_hitting_nodes('barabasi', no_of_nodes, start_node, no_of_steps, 1)
        bars2 = avg_hitting_nodes('watts', no_of_nodes, start_node, no_of_steps, 1)
        bars3 = avg_hitting_nodes('random', no_of_nodes, start_node, no_of_steps, 1)
        Bars1.append(bars1)
        Bars2.append(bars2)
        Bars3.append(bars3)
    data = []
    for i in range(len(Bars1)):
        data.append(['barabasi', Bars1[i][node], node])
        data.append(['watts', Bars2[i][node], node])
        data.append(['random', Bars3[i][node], node])
    df = pd.DataFrame(data, columns=['type', 'avg', 'node'])
    print(df.head())
    pal = sns.color_palette(palette='muted')
    g = sns.FacetGrid(df, row='type', hue='node', aspect=15, height=0.75, palette=pal)
    g.map(sns.kdeplot, 'avg', bw_adjust=1.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, 'avg', bw_adjust=1.5, clip_on=False, color="w", lw=2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    for i, ax in enumerate(g.axes.flat):
        ax.text(50, 0.02, dictionary[i + 1],
                fontweight='bold', fontsize=15,
                color=ax.lines[-1].get_color())
    g.fig.subplots_adjust(hspace=-0.1)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    plt.xlabel('Average hits of node {} for starting node {}'.format(node, start_node), fontweight='bold', fontsize=15)
    plt.show()

plot_of_avg_hits_of_specific_node(10, 0, 50, 1, 50)
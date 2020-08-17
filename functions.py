import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
from scipy.special import comb

random.seed(0)

""" The return values in all functions are only for demonstrating the format, and should be overwritten """


class ModelData:
    def __init__(self, dataset):
        self.init_graph = pd.read_csv(dataset)
        self.nodes = self.init_graph[['source', 'target']]


def graphA(links_data):
    g = nx.from_pandas_edgelist(links_data.nodes, 'source', 'target')
    return g


def calc_best_power_law(G):
    y = nx.degree_histogram(G)
    x=[]
    for h in range(len(y)):
        if y[h] != 0:
            x = x+[h]

    y[:] = (value for value in y if value != 0)
    x = np.array(x)
    y = np.array(y)
    x_data = np.log(x)
    y_data = np.log(y)

    curve_fit = np.polyfit(x_data, y_data,1)
    alpha = -(curve_fit[0])
    beta = np.exp(curve_fit[1])

    return alpha, beta


def plot_histogram(G, alpha, beta):
    y = nx.degree_histogram(G)
    del y[0]
    y = np.array(y)
    x = np.arange(1, len(y) + 1, 1)
    y_fit = (x**-alpha)*beta

    plt.plot(x, y, 'o')
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, y_fit)
    plt.show()

    return


def G_features(G): #########return dict of dicts ########
    dict1 = nx.closeness_centrality(G)
    dict2 = nx.betweenness_centrality(G)

    #return {'Closeness': {1: 0, 2: 0, 3: 0}, 'Betweeness': {1: 0, 2: 0, 3: 0}}
    return{'Closeness': dict1, 'Betweeness': dict2}

def create_undirected_weighted_graph(links_data, users_data, question):  #########return Graph ########

    WG = nx.from_pandas_edgelist(links_data.init_graph, 'source', 'target', 'weight')
    nodesList = list(users_data['node'])
    infected = list(users_data[question])
    i = 0
    while i < len(nodesList):
        WG.nodes[nodesList[i]]['infected'] = infected[i]
        i += 1

    return WG

def run_k_iterations(WG, k, Threshold):
    exposure=0
    listt=[]
    S = dict()
    infected = list()
    nodesList=list(WG.nodes)
    for i in nodesList:
        if WG.nodes[i]['infected'] == 'YES':
            infected.append(i)
    S[0]=infected
    m = 1
    while m <= k:
        infected = []
        i = 0
        while i < len(WG.nodes):
            if WG.nodes[nodesList[i]]['infected'] == 'YES':
                i += 1
                continue
            for j in list(WG.neighbors(nodesList[i])):
                if WG.nodes[j]['infected'] == 'YES':
                    exposure += WG.edges[nodesList[i], j]['weight']

                if exposure >= Threshold:
                    infected.append(nodesList[i])
                    break
            i += 1
            exposure = 0
        for inf in infected:
            WG.nodes[inf]['infected'] = 'YES'
        if infected:
            S[m]=infected
        m += 1

    return S

# return a dictionary of with the branching factors R1,...Rk
def calculate_branching_factor(S, k):
    branching_fac= dict()
    for i in range(k+1):
        if i==0: continue
        val=len(S[i])/len(S[i-1])
        branching_fac[i]=val

    return branching_fac

# return a dictionary of the h nodes with the highest degree, sorted by decreasing degree
# {index : node}
def find_maximal_h_deg(WG, h):
    nodes_dict = {}
    temp_nodes= dict()
    j = 1
    nodesList=list(WG.nodes)
    for i in nodesList:
        temp_nodes[i]=WG.degree[i]

    for j in range(h):
        tempMax=max(temp_nodes, key=temp_nodes.get)
        del temp_nodes[tempMax]
        nodes_dict[tempMax]=WG.degree[tempMax]

    return nodes_dict


# return a dictionary of all nodes with their clustering coefficient
# {node : CC}
def calculate_clustering_coefficient(WG, nodes_dict):
    nodes_dict_clustering={}
    for node in nodes_dict.keys():
        neighbours = [n for n in nx.neighbors(WG, node)]
        n_neighbors = len(neighbours)
        n_links = 0
        if n_neighbors > 1:
            for node1 in neighbours:
                for node2 in neighbours:
                    if WG.has_edge(node1, node2):
                        n_links += 1
            clustering_coefficient = n_links / (0.5 * n_neighbors * (n_neighbors - 1))
            nodes_dict_clustering[node]=clustering_coefficient
        else:
            nodes_dict_clustering[node]=0
    return nodes_dict_clustering


def infected_nodes_after_k_iterations(WG, k, Threshold):
    infnodes = run_k_iterations(WG, k, Threshold)
    count = 0
    for x in infnodes:
        if isinstance(infnodes[x], list):
            count += len(infnodes[x])
    return count


# return the first [number] nodes in the list
def slice_dict(dict_nodes, number):
    nodes_list = list(dict_nodes.keys())
    nodes_list[:] = (value for value in nodes_list if nodes_list.index(value) < number)

    return nodes_list


# remove all nodes in [nodes_list] from the graph WG, with their edges, and return the new graph
def nodes_removal(WG, nodes_list):
    for i in range(len(nodes_list)):
        WG.remove_node(nodes_list[i])
    return WG


# plot the graph according to Q4 , add the histogram to the pdf and run the program without it
def graphB(number_nodes_1, number_nodes__2, number_nodes_3):

    lists = sorted(number_nodes_1.items())
    x1, y1 = zip(*lists)

    lists = sorted(number_nodes__2.items())
    x2, y2 = zip(*lists)

    lists = sorted(number_nodes_3.items())
    x3, y3 = zip(*lists)

    plt.plot(x1, y1,'-.')
    plt.plot(x2, y2,':')
    plt.plot(x3, y3,'--')
    plt.xlabel('Removed nodes')
    plt.ylabel('Infected nodes')
    plt.legend('abc')
    plt.show()
    return


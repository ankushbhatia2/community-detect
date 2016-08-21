#-------------------------------------------------------------------------------
# Name:        community-detect.py
# Purpose:     A community detection module based on my research paper :
#              "Community detection for personalization in networks based
#              on structural and persona similarities"
#
# Author:      Ankush Bhatia
#
# Created:     28/06/2016
# Copyright:   <add copyright>
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import networkx as nx

from itertools import permutations
from itertools import combinations
from collections import defaultdict

import matplotlib.pyplot as plt
import random

class Community(object):
    def __init__(self, alpha_weight = 0.5):
        self.alpha = alpha_weight
        random.seed()
        self.MIN_VALUE = 0.0000001
        self.node_weights = {}

    @classmethod
    # Louvain Modularity implementation based on https://github.com/shogo-ma/louvain-python
    # Author : Shogo Ma
    # Date : 25th Nov 2015
    # Name : 'louvain-python'
    # Version : "0.0.1"
    #To convert an IGraph to NetworkX Graph
    def convertIGraphToNxGraph(cls, igraph):
        nodenames = igraph.vs["name"]
        edges = igraph.get_edgelist()
        weights = igraph.es["weight"]
        nodes = defaultdict(str)

        for idx, node in enumerate(igraph.vs):
            nodes[node.index] = nodenames[idx]

        convertlist = []
        for idx in range(len(edges)):
            edge = edges[idx]
            newedge = (nodes[edge[0]], nodes[edge[1]], weights[idx])
            convertlist.append(newedge)

        convert_graph = nx.Graph()
        convert_graph.add_weighted_edges_from(convertlist)
        return convert_graph

    #Updating Nodes weight
    def updateNodeWeights(self, edgeweights):
        nodeweights = defaultdict(float)
        for node in edgeweights.keys():
            nodeweights[node] = sum([weight for weight in edgeweights[node].values()])
        return nodeweights

    #Main function of Louvain Algorithm
    def getPartition(self, graph, param=1.):
        node2com, edge_weights = self._setNode2Com(graph)
        node2com = self.runFirstPhase(node2com, edge_weights, param)
        best_modularity = self.ComputeModularity(node2com, edge_weights, param)
        partition = node2com.copy()
        new_node2com, new_edge_weights = self.runSecondPhase(node2com, edge_weights)
        while True:
            new_node2com = self.runFirstPhase(new_node2com, new_edge_weights, param)
            modularity = self.ComputeModularity(new_node2com, new_edge_weights, param)
            if abs(best_modularity - modularity) < self.MIN_VALUE:
                break
            best_modularity = modularity
            partition = self._updatePartition(new_node2com, partition)
            new_node2com_, new_edge_weights_ = self.runSecondPhase(new_node2com, new_edge_weights)
            new_node2com = new_node2com_
            new_edge_weights = new_edge_weights_
        return partition

    #Newmann Modularity
    def ComputeModularity(self, node2com, edge_weights, param):
        q = 0
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)
        for com_id, nodes in com2node.items():
            node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes]
            cluster_weight = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations])
            tot = self.getDegreeOfCluster(nodes, node2com, edge_weights)
            q += (cluster_weight / (2 * all_edge_weights)) - param * ((tot / (2 * all_edge_weights)) ** 2)
        return q

    def getDegreeOfCluster(self, nodes, node2com, edge_weights):
        weight = sum([sum(list(edge_weights[n].values())) for n in nodes])
        return weight

    def _updatePartition(self, new_node2com, partition):
        reverse_partition = defaultdict(list)
        for node, com_id in partition.items():
            reverse_partition[com_id].append(node)

        for old_com_id, new_com_id in new_node2com.items():
            for old_com in reverse_partition[old_com_id]:
                partition[old_com] = new_com_id
        return partition

    def runFirstPhase(self, node2com, edge_weights, param):
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        self.node_weights = self.updateNodeWeights(edge_weights)
        status = True
        while status:
            statuses = []
            for node in node2com.keys():
                statuses = []
                com_id = node2com[node]
                neigh_nodes = [edge[0] for edge in self.getNeighborNodes(node, edge_weights)]

                max_delta = 0.
                max_com_id = com_id
                communities = {}
                for neigh_node in neigh_nodes:
                    node2com_copy = node2com.copy()
                    if node2com_copy[neigh_node] in communities:
                        continue
                    communities[node2com_copy[neigh_node]] = 1
                    node2com_copy[node] = node2com_copy[neigh_node]

                    delta_q = 2 * self.getNodeWeightInCluster(node, node2com_copy, edge_weights) - (self.getTotWeight(
                        node, node2com_copy, edge_weights) * self.node_weights[node] / all_edge_weights) * param
                    if delta_q > max_delta:
                        max_delta = delta_q
                        max_com_id = node2com_copy[neigh_node]

                node2com[node] = max_com_id
                statuses.append(com_id != max_com_id)

            if sum(statuses) == 0:
                break

        return node2com

    def runSecondPhase(self, node2com, edge_weights):
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda: defaultdict(float))

        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
        for edge in node_pairs:
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][
                edge[1]]
        return new_node2com, new_edge_weights

    def getTotWeight(self, node, node2com, edge_weights):
        nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    def getNeighborNodes(self, node, edgeweights):
        if node not in edgeweights:
            return 0
        return edgeweights[node].items()

    def getNodeWeightInCluster(self, node, node2com, edge_weights):
        neigh_nodes = self.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]
        return weights

    def _setNode2Com(self, graph):
        node2com = {}
        edge_weights = defaultdict(lambda: defaultdict(float))
        for idx, node in enumerate(graph.nodes()):
            node2com[node] = idx
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]["weight"]
        return node2com, edge_weights


    def make_k_nearest_neighbour_graph(self, Graph, vertices, k, similarity_matrix, similarity_matrix_type='cosine'):
        #Directed Graph
        knng = nx.DiGraph()

        #Iteration over all vertices
        for i in vertices:

            #Similarity list of i with all other vertices
            sim_i = []
            for j in range(len(similarity_matrix[i])):
                if j != i:
                    if similarity_matrix_type == 'cosine':
                        g = 0
                    else:
                        g = 1
                    if j in Graph[i]:
                        g = abs(g-1)
                    sim = self.alpha * g + (1 - self.alpha) * float(similarity_matrix[i][j])
                    sim_i.append((sim, j))
            sim_i.sort(reverse=True)
            for j in range(k):
                # print (sim_i[j][1])
                knng.add_edge(int(i), int(sim_i[j][1]), weight=sim_i[j][0])
        return knng

    def get_communities(self, Graph, vertices, similarity_matrix, similarity_matrix_type='cosine'):
        #try:
            #Total Edges
            m = len(Graph.edges())
            # Setting k for knn graph
            k = (m // len(vertices))
            #Making a k-nearest-neighbour-graph
            knng = self.make_k_nearest_neighbour_graph(Graph=Graph, vertices=vertices, k=k,
                                                   similarity_matrix=similarity_matrix,
                                                   similarity_matrix_type=similarity_matrix_type)

            #Getting communities
            partition = self.getPartition(knng)
            communities = defaultdict(list)
            for node, com_id in partition.items():
                communities[com_id].append(node)
            return communities
        #except Exception as err:
        #    print(err)

    #Requires Matplotlib
    def view_communities(self, communities, Graph, vertices, similarity_matrix, similarity_matrix_type):
        try:
            # Total Edges
            m = len(Graph.edges())
            # Setting k for knn graph
            k = (m // len(vertices))
            # Making a k-nearest-neighbour-graph
            knngraph = self.make_k_nearest_neighbour_graph(Graph=Graph, vertices=vertices, k=k,
                                                       similarity_matrix=similarity_matrix,
                                                       similarity_matrix_type=similarity_matrix_type)
            #Viewing Graph
            pos = nx.spring_layout(knngraph)
            red_edges = []
            blue_edges = []

            colors = ['r', 'b', 'g', '#FF0099', '#660066', '#FFFFFF', '#000000', '#123456', '#00FFFF', '#A056F2', '#888888',
                 '#AABBCC',
                 '#BFCFDF', '#500000', '#EFFEEF']
            i = 0
            for com, nodes in communities.items():
                nx.draw_networkx_nodes(knngraph, pos, cmap=plt.get_cmap('jet'), nodelist=nodes,
                                   node_size=100, node_color=colors[(i-1)%len(communities)])
                i+=1

            for edge in knngraph.edges():
                found = False
                for com, nodes in p.items():
                    if edge[0] in nodes and edge[1] in nodes:
                        blue_edges.append(edge)
                        found = True
                if found == False:
                    red_edges.append(edge)

            nx.draw_networkx_edges(knngraph, pos, edgelist=red_edges, edge_color='r', arrows=True)
            nx.draw_networkx_edges(knngraph, pos, edgelist=blue_edges, edge_color='b', arrows=True)
            return plt
        except Exception as err:
            print(err)


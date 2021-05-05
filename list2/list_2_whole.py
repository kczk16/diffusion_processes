class Graph:
    '''
    A class used to represent a graph.
    '''

    def __init__(self, vertices = [], edges = [], weights = {}):
        '''
        :param vertices: list of vertices in a graph i.e.: ['a', 'b', 'c']
        :type vertices: list
        :param edges: list of edges in a graph i.e.: [['a','b'], ['b','c']]
        :type edges: list
        :param weights: dictionary which keys are tuples of edges of the graph and values
                        are weights of those edges i.e.:{('a','b') : 1, ('b','c') : 1}
        :type weights: dict
        '''
        self.vertices = vertices
        self.edges = edges
        self.weights = weights


    def addVertex(self, vert):
        '''
        Method that adds a vertex to the graph.
        :param vert: vertex to be added to the graph, i.e.: 'd'
        :type vert: str
        :raises:
         Exeption: if vert already in the graph
        '''
        if vert not in self.vertices:
            self.vertices.append(vert)
        else:
            raise Exception("vertex already in graph")


    def addVerticesFromList(self, vertList):
        '''
        Method that adds vertices from list to the graph.
        :param vertList: list of vertices to be added to the graph, i.e.: ['a','b']
        :type vertList: list
        '''
        for vert in vertList:
            if vert not in self.vertices:
                self.vertices.append(vert)
            else:
                continue


    def addEdge(self, fromVert, toVert, weight = None):
        '''
        Method that adds edge between vertices to the graph.
        :param fromVert: edge start vertex, i.e.: 'a'
        :type fromVert: str
        :param toVert: edge end vertex, i.e.: 'b'
        :type toVert: str
        :param weight: weight of the edge to be added, i.e.: 2
        :type weight: int
        '''
        if [fromVert, toVert] and [toVert, fromVert] not in self.edges:
            self.edges.append([fromVert, toVert])
        if fromVert not in self.vertices:
            self.vertices.append(fromVert)
        if toVert not in self.vertices:
            self.vertices.append(toVert)
        if weight is not None:
            if [toVert, fromVert] not in self.weights:
                self.weights[fromVert, toVert] = weight
            else:
                self.weights[toVert, fromVert] = weight


    def addEdgesFromList(self, edgeList):
        '''
        Method that adds edges from list to the graph.
        :param edgeList: list of edges to be added, i.e.: [['a','b'], ['b', 'c']]
        :type edgeList: list
        '''
        for edge in edgeList:
            if edge[:2] not in self.edges:
                self.edges.append(edge[:2])
                if edge[0] not in self.vertices:
                    self.vertices.append(edge[0])
                if edge[1] not in self.vertices:
                    self.vertices.append(edge[1])
            try:
                self.weights[tuple(edge[:2])] = edge[2]
            except IndexError as error:
                continue

    def getVertices(self):
        '''
        Method that gets vertices of the graph.
        :return: vertices
        :rtype: list of str
        '''
        return self.vertices


    def getEdges(self):
        '''
        Method that gets edges of the graph.
        :return: edges
        :rtype: list of lists
        '''
        return self.edges


    def getNeighbors(self, vertKey):
        '''
        Method that gets neighbors of a vertex.
        :param vertKey: vertex of which neighbors will be returned, i.e.: 'a'
        :type vertKey: str
        :return: Neighbors
        :rtype: list
        '''
        Neighbors = []
        for edge in self.edges:
            if vertKey in edge:
                index = edge.index(vertKey)
                if index:
                    Neighbors.append(edge[0])
                else:
                    Neighbors.append(edge[1])
        return Neighbors


    def __contains__(self, item):
        if item in self.vertices:
            return True
        else:
            return False


    def saveGraph(self):
        '''
        Method that saves dot representation of the graph to a text file.
        '''
        for edge in self.edges:
            if tuple(edge[:2]) not in self.weights.keys():
                self.weights[tuple(edge[:2])] = 1
        with open('graph.txt', 'w') as text:
            text.write('graph {\n')
            for edge in self.edges:
                start = edge[0]
                stop = edge[1]
                text.write('{} -- {} [weight={}]\n'.format(start, stop, self.weights[(start, stop)]))
            text.write('}')


    def getShortestPaths(self, fromVert):
        '''
        Method that calculates shortest paths in the graph from the given vertex
        to all other vertices using Dijkstra's algorithm.
        :param fromVert: edge start vertex, i.e.: 'a'
        :type fromVert: str
        :return: shortest_distance
        :rtype: dict
        '''
        for edge in self.edges:
            if tuple(edge[:2]) not in self.weights.keys():
                self.weights[tuple(edge[:2])] = 1
        
        start = fromVert
        prev_dist = 0
        unseenNodes = self.getVertices()
        w = self.weights
        n = self.getNeighbors(start)
        
        shortest_distance = {}
        distance = {}

        for node in unseenNodes:
            shortest_distance[node] = float("inf")
            distance[node] = float("inf")
        shortest_distance[start] = 0 
        distance[start] = 0 

        while len(unseenNodes) > 1:
            for node in n: 
                if node in unseenNodes:
                    k = (w.get((start, node)) or w.get((node, start)))
                    di = k + prev_dist
                    if di < shortest_distance[node]:
                        shortest_distance[node] = di
                        distance[node] = di
     
            del distance[start]        
            next_vert = min(distance, key=distance.get)
            start = next_vert
            prev_dist = shortest_distance[start]
            unseenNodes.remove(start)
            n = self.getNeighbors(start)
            
        return shortest_distance
            


graph = Graph()
#graph.addEdgesFromList([['a', 'b', 7], ['b', 'c', 1], ['a', 'c', 3], ['d', 'c', 2], ['b', 'd', 2], ['b', 'e', 6],
# ['d', 'e']])
graph_from_1 = [['A', 'B'], ['C', 'A'], ['A', 'D'], ['A', 'E'], ['A', 'F'], ['B', 'G'], ['G', 'H'], ['H', 'J'],
                ['J', 'G'], ['H', 'I'], ['I', 'G'], ['I', 'J'], ['E', 'F'], ['D', 'C'], ['C', 'F']]
graph.addEdgesFromList(graph_from_1)
print(graph.getShortestPaths('C'))
graph.saveGraph()

graph.vertices = ['a', 'b', 'c']
graph.edges = [['a', 'b']]
graph.addVertex('d')
graph.addEdge(fromVert='a', toVert='c')
graph.addEdgesFromList([['a', 'e'], ['b', 'c', 2]])
graph.addVerticesFromList(['d', 'f'])
n = graph.getNeighbors('a')
e = graph.getEdges()
v = graph.getVertices()
w = graph.weights
print(n, 'neighbors')
print(e, 'edges')
print(v, 'vertices')
print(w, 'weights')










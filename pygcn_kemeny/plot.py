import matplotlib.pyplot as plt
import networkx as nx #I have version 2.3
'''
graph = {
    '1': ['2', '3', '4'],
    '2': ['5','11','12','13','14','15'],
    '3': ['6','7','66','77'],
    '5': ['6', '8','66','77'],
    '4': ['7','66','77'],
    '7': ['9', '10']
    }

MG = nx.DiGraph(graph)
print (MG)
exit()

plt.figure(figsize=(8,8))
pos=nx.graphviz_layout(MG,prog="twopi",root='1')

nodes = MG.nodes()
degree = MG.degree()
color = [degree[n] for n in nodes]
size = [2000 / (degree[n]+1.0) for n in nodes]

nx.draw(MG, pos, nodelist=nodes, node_color=color, node_size=size,
        with_labels=True, cmap=plt.cm.Blues, arrows=False)
plt.show()
'''

G = nx.DiGraph()
G.add_nodes_from([1,2,3,4])
G.add_edges_from([(1,2, {'weight': 0.5}), (1,3, {'weight': 0.5}), 
	(2,1, {'weight': 1/3}), (2,3, {'weight': 1/3}), (2,4, {'weight': 1/3}),
	(3,1, {'weight': 0.5}), (3,2, {'weight': 0.5}),
	(4,2, {'weight': 1})])

weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.draw(G, with_labels=True, width=weights)
plt.show()


#Graph analysis
#print (list(nx.connected_components(G)))    #not implemented for DiGraph
print (nx.clustering(G))
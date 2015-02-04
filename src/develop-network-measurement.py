import networkx as nx
import numpy as np 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import Graphics as artist

from awesome_print import ap 
from matplotlib import rcParams

rcParams['text.usetex'] = True

def ecdf(arr):
	sorted_arr = np.sort(arr)
	return (sorted_arr,np.arange(len(sorted_arr))/float(len(sorted_arr)))

def degree(graph):
	return nx.degree(graph).values()

nodes = 1000
total_edges = 400
average_edges = 5
average_edges /= float(nodes)

random_graph = nx.erdos_renyi_graph(nodes,average_edges)
structured_graph = nx.gnm_random_graph(nodes,total_edges)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.hold(True)
x,y = ecdf(degree(random_graph))
ax.plot(x,y)
del x,y

x,y = ecdf(degree(structured_graph))
ax.plot(x,y)
del x,y

plt.tight_layout()
plt.show()
#ap(nx.degree(random_graph))

#ap(nx.degree(complete_graph))

import matplotlib.pyplot as plt
import networkx as nx
import os
from random import randrange
import numpy

def obtainData(fname):
    results_dir = createFolder(fname)
    edges = readGraph(fname)
    G = nx.Graph()
    G.add_edges_from(edges)
    adjacency_matrix(G,results_dir)
    avg_clustering(G,results_dir)
    plot_degree_dist(G,results_dir, fname)
    sampleRandomNode(G, results_dir)


def createFolder(fname):
    #creation of the results folder
    script_dir = os.path.dirname(__file__)
    string = fname.split('.',1)[0]
    dir = 'Results/' + string + '/'
    results_dir = os.path.join(script_dir, dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    return results_dir 
    
def readGraph(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            data.append(lineToData(line.split()))
    return data
    
def lineToData(line):
    return(int(line[0]),int(line[1]))

def adjacency_matrix(G, results_dir):
    matrix = nx.adjacency_matrix(G)
    matrix = matrix.todense()
    print(matrix)
    mat = numpy.asarray(matrix)
    fname = results_dir + 'adjacency.csv'
    numpy.savetxt(fname, mat, delimiter=",")

def avg_clustering(G, results_dir):
    clustering = nx.average_clustering(G)
    fname = results_dir + 'clustering.txt'
    with open(fname,'w') as f:
        string =  f'The average clustering value is {clustering}'
        f.write(string)

def plot_degree_dist(G, results_dir, fname):
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    degree_normalized = []
    num = G.number_of_nodes()
    print(degrees)
    for x in degree_freq:
        degree_normalized.append(x/num)
    # print(degree_normalized)
    plt.title(f'Histogram for the normalized values of the degree for graph {fname}')
    plt.hist(degree_normalized)
    histname = results_dir + 'histogram.svg' #TODO: Save fig 
    plt.savefig(histname)
    plt.close()
    m = 3
    plt.figure(figsize=(12, 8)) 
    plt.loglog(degrees[m:], degree_normalized[m:],'g-') 
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Probability mass function for graph {fname}')
    logname = results_dir + 'loglog.svg'
    plt.savefig(logname)


def sampleRandomNode(G, results_dir):
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    avg_degree = 0
    for deg in degrees:
        avg_degree += deg * degree_freq[deg]
    avg_degree = avg_degree / G.number_of_nodes()
    counter = 0
    ran = 1000
    for iter in range(0,ran):
        num = randrange (0, G.number_of_nodes() + 1)
        deg = G.degree[num]
        if deg < avg_degree:
            counter = counter + 1
    fname = results_dir + 'sampleRandom.txt'
    with open(fname,'w') as f:
        string =  f'out of {ran} trials, we found a lower number {counter} times'
        f.write(string)
   
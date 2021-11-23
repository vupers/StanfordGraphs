import matplotlib.pyplot as plt
import networkx as nx
import os
from random import randrange
from networkx.classes.function import neighbors
import numpy
import seaborn as sns
import bonusPart
def obtainData(fname):
    results_dir = createFolder(fname)
    edges = readGraph(fname)
    G = nx.Graph()
    G.add_edges_from(edges)
    adjacency_matrix(G,results_dir)
    avg_clustering(G,results_dir)
    plot_degree_dist(G,results_dir, fname)
    sampleRandomNode(G, results_dir)
    bonusPart.estimateValues(G, results_dir)
    bonusPart.poissonFit(G)

def createFolder(fname):
    """creation of the results folder"""
    print ('Creating folder...\n')
    script_dir = os.path.dirname(__file__)
    string = fname.split('.',1)[0]
    dir = 'Results/' + string + '/'
    results_dir = os.path.join(script_dir, dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    return results_dir 
    
def readGraph(fname):
    """Read the data of the graph from the text file"""
    print ('Reading file...\n')
    data = []
    with open(fname) as f:
        for line in f.readlines():
            data.append(lineToData(line.split()))
    print(f'{fname} was succesfully read!\n')
    return data
    
def lineToData(line):
    """Reads a line and returns the indices of two nodes"""
    return(int(line[0]),int(line[1]))

def adjacency_matrix(G, results_dir):
    """ii) Determine the adjacency matrix A."""
    print ('Determining adjacency matrix...\n')

    matrix = nx.adjacency_matrix(G)
    print(matrix)
    # matrix = matrix.todense()
    # print(matrix)
    # mat = numpy.asarray(matrix)
    # fname = results_dir + 'adjacency.csv'
    #numpy.savetxt(fname, mat, delimiter=",")

def avg_clustering(G, results_dir):
    """iii) Calculate the average clustering coefficient of each network"""
    print ('Determining average clustering...\n')
    clustering = nx.average_clustering(G)
    fname = results_dir + 'clustering.txt'
    with open(fname,'w') as f:
        string =  f'The average clustering value is {clustering}'
        f.write(string)

def plot_degree_dist(G, results_dir, fname):
    """Estimate the degree distribution of the networks and plot the probability mass
    functions pm (pm := P(K = m), where K is the degree of the node). Plot a log-log
    plot as well."""
    #generate normalized values for y axis and range of values for x axis
    print ('Generating graphs...\n')
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    degree_normalized = []
    num = G.number_of_nodes()
    for x in degree_freq:
        degree_normalized.append(x/num)

    #bar plot
    plt.title(f'Degree distribution for graph {fname}')
    plt.bar(degrees, degree_normalized)
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    histname = results_dir + 'histogram.svg' 
    plt.savefig(histname)
    plt.close()

    #loglog graph
    m = 3
    plt.figure(figsize=(12, 8)) 
    plt.loglog(degrees[m:], degree_normalized[m:],'g-') 
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Probability mass function for graph {fname}')
    logname = results_dir + 'loglog.svg'
    plt.savefig(logname)
    plt.close()


def sampleRandomNode(G, results_dir):
    """v) Calculate the average degree of the neighbors of a randomly chosen node in one
    network of your choice. Compare the result with the average degree of the network.
    Can you observe the friendship paradox, i.e. on average, your friends have more
    friends than you do?"""
    print ('Ordering some McDonalds...\n')

    avg_degree = calculateAvgDegree(G)

    #sample 1000 nodes to check for their degree
    counter = 0
    testSize = 1000
    valuesList = []
    for _ in range(0,testSize):
        num = randrange (0, G.number_of_nodes())
        if G.has_node(num):
            #print(f'{num = }')
            deg = G.degree[num]
            valuesList.append(deg)
            if deg < avg_degree:
                counter = counter + 1
        else:
            print('Broken node :(')
    fname = results_dir + 'sampleRandom.txt'
    with open(fname,'w') as f:
        string1 = f'The average degree of the graph is {avg_degree}\n'
        string2 =  f'out of {len(valuesList)} valid trials, we found a lower number {counter} times\n'
        f.write(string1)
        f.write(string2)
        for value in valuesList:
            f.write(f'{value}\n')

def calculateAvgDegree(G):
    """calculate the average degree of the network G. 
        Parameter: NXgraph
        Returns: float
        """
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    avg_degree = 0
    for deg in degrees:
        avg_degree += deg * degree_freq[deg]
    avg_degree = avg_degree / G.number_of_nodes()
    return avg_degree
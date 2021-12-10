import networkx as nx
import matplotlib.pyplot as plt
import random
from math import factorial,exp
import numpy as np
from scipy.special import binom
from scipy import sparse
from scipy.stats import poisson
def wattsStrogatz(n,r,p):
    G = nx.Graph()
    for node in range (0,n):
        #generate neighboring nodes
        for neighborIndex in range(1,r+1):
            adjacentNode = (neighborIndex + node) % (n)
            G.add_edge(node, adjacentNode)

    listOfLists=[]
    for node in range(0,n):
        neighborList = []
        for adjacentNode in G.neighbors(node):
            #HANDLING THE BOUNDARY CASES
            if (node - r < 0):
                if adjacentNode - node <= r and adjacentNode - node > 0:
                    neighborList.append(adjacentNode)
            elif (node + r >= n):
                if adjacentNode - node > 0 or node - adjacentNode > r:
                    neighborList.append(adjacentNode)
            else:
                if (adjacentNode > node):
                    neighborList.append(adjacentNode)
        # print(neighborList)
        listOfLists.append(neighborList)

    for node in range(0,n):
        updatedList = listOfLists[node]
        for adjacentNode in updatedList:
            probability = random.uniform(0, 1)
            if probability <= p:
                connection = node
                while (connection == node or
                connection in G.neighbors(node)
                or connection == n):
                    connection = random.randint(0,n)
                G.remove_edge(node,adjacentNode)
                G.add_edge(node,connection)
                # print(f"Deleted [{node} - {adjacentNode}]")
                # print(f"Rewired [{node} - {connection}]")
                # updatedList.remove(adjacentNode)
                # updatedList.append(connection)

    return G

def actualDistribution(m,r,p):
    minValue = 0
    if (m - r <= r):
        minValue = m - r
    else: 
        minValue = r
    massProbabilityFunction = 0
    for n in range (0, minValue):
        term = binom(r,n)*pow(1-p,n) * pow(p,r-n) * pow(r*p ,m-r-n) * exp(-p*r) / factorial(m-r-n) 
        massProbabilityFunction += term

    return massProbabilityFunction


def main():
    #expected average of the erdos renyi model: n*p
    # 2*r = n* p -> p = 2*r / n
    n = 500
    r = 10
    p = 1
    G = wattsStrogatz(n=n, r = r, p = p)

    newP = 2 * r / n
    G2 = nx.erdos_renyi_graph(n,newP)

    degree_freq = nx.degree_histogram(G)
    degrees = np.arange(len(degree_freq))
    degree_normalized = []
    num = G.number_of_nodes()
    #print(G.degree)
    for x in degree_freq:
        degree_normalized.append(x/num)

    erdos_freq = nx.degree_histogram(G2)
    degrees2 = np.arange(len(erdos_freq))
    erdos_normalized = []
    num = G2.number_of_nodes()
    #print(G.degree)
    for x in erdos_freq:
        erdos_normalized.append(x/num)


    valueList = []
    valueList2 = []
    for m in degrees:
            valueList2.append(actualDistribution(m, r = r, p = 0.999))
            if(m-r>= 0):
                valueList.append(poisson.pmf(m-r,r))
            else:
                valueList.append(0)
            
            
    # print(valueList2)
    # print(valueList)
    # print(sum(valueList2))
    # print(sum(valueList))

    cut=150
    degrees = degrees[cut:]
    degrees2 = degrees2[cut:]
    degree_normalized = degree_normalized[cut:]
    valueList = valueList[cut:]
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Watts-Strogatz degree distribution for {n=},{r=},{p=}')
    plt.bar(degrees,degree_normalized)
    plt.show()

    
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Watts-Strogatz degree distribution for {n=},{r=},{p=}')
    ax = plt.subplot(111)
    ax.bar(degrees-0.2, degree_normalized, width=0.25, color='b', align='center', label='Empirical Distribution')
    ax.bar(degrees, valueList2, width=0.25, color='y', align='center', label='Poisson Distribution for p->1')
    ax.bar(degrees2 + 0.2, erdos_normalized, width=0.25, color='r', align='center',label = 'Erdos Distribution. n=500, p = 2*r / n')
    plt.legend(loc='upper left')
    plt.show()

    
    # plt.bar(degrees,degree_normalized, alpha=0.5)
    # plt.bar(degrees,valueList, alpha = 0.9)
    #plt.bar([degree_normalized, valueList],label=['Empirical Distribution','Actual Distribution'])  
    #plt.legend(loc='upper right')
    
    # plt.show()
    # pos = nx.circular_layout(G)
    # plt.figure(figsize = (12, 12))
    # nx.draw_networkx(G, pos)
    # plt.show()

if __name__ == '__main__':
    main()
    # m = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # c = np.array([0,1,2])
    # print(sparse.csr_matrix(m).multiply(sparse.csr_matrix(c)).todense())

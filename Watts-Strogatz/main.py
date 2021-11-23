import networkx as nx
import matplotlib.pyplot as plt

def main():
    G = nx.watts_strogatz_graph(n=500, k = 5, p = 0.2, seed=42)
    # pos = nx.circular_layout(G)
    # plt.figure(figsize = (12, 12))
    # nx.draw_networkx(G, pos)
    # plt.show()
    # plt.close()
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    degree_normalized = []
    num = G.number_of_nodes()
    for x in degree_freq:
        degree_normalized.append(x/num)
    plt.title(f'Degree distribution for graph G')
    plt.bar(degrees, degree_normalized)
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.show()

if __name__ == '__main__':
    main()
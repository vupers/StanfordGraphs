from random import randrange
import matplotlib.pyplot as plt
from scipy.stats import poisson, powerlaw, chisquare
from sklearn.metrics import r2_score
import networkx as nx


def estimateValues(G, results_dir):
    print ('Estimating values...\n')

    iterations = 100000
    validIterations=0
    counter = 0
    sumOfKv0 = 0
    sumOfKv1 = 0
    for _ in range(0,iterations):
        randomNode = randrange (0, G.number_of_nodes())
        if not G.has_node(randomNode):
            #   print('randomNode is broken :((')
            continue
        numberOfNeighbors = sum(1 for _ in G.neighbors(randomNode))
        neighborRange = randrange(0, numberOfNeighbors)
        neighborIndex = -1
        neighborCounter = 0
        for neighbor in G.neighbors(randomNode):
            if neighborCounter == neighborRange:
                neighborIndex = neighbor
            neighborCounter +=1
        if not G.has_node(neighborIndex):
            #print('neighborIndex is broken :((')
            continue
        Kv0 = G.degree[randomNode]
        Kv1 = G.degree[neighborIndex]
        sumOfKv0 += Kv0
        sumOfKv1 += Kv1
        validIterations+=1
        # print(f'v0 is node {randomNode}. v1 is node {neighborIndex}.')
        # print(f'Kv0 = {Kv0}, Kv1 = {Kv1}')
        if Kv1 > Kv0:
            counter = counter + 1
    expectedValueKv0 = sumOfKv0 / validIterations
    expectedValueKv1 = sumOfKv1 / validIterations

    fname = results_dir + 'estimateValues.txt'
    with open(fname,'w') as f:
        string1 = f'E(Kv0) = {expectedValueKv0}, E(Kv1) = {expectedValueKv1}\n'
        string2 =  f'out of {validIterations} trials, Kv1 was larger than Kv0 {counter} times'
        f.write(string1)
        f.write(string2)

def poissonFit(G):
    #obtain lambda for poisson using MLE
    """ the estimator lambda_n is just the sample mean of the n observations in the sample.
    References: https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood
    https://en.wikipedia.org/wiki/Poisson_distribution#Parameter_estimation"""
    print ('Determining value of lambda...\n')
    testRange= 100_000
    degreeSum = 0
    degrees = []
    for _ in range(0,testRange):
        randomNode = randrange (0, G.number_of_nodes())
        if not G.has_node(randomNode):
            #print('[POISSON]randomNode is broken :((')
            continue
        degreeSum += G.degree[randomNode]
        degrees.append(G.degree[randomNode])

    lambdaEstimate = degreeSum / len(degrees)
    print(f'Using MLE, we obtain a value of {lambdaEstimate = }.')

    # #generate the list that contains all the degrees of the graph
    # values = []
    # for node in list(G.nodes):
    #     values.append(G.degree[node])

    # #generate a list of random values following the poisson distribution with the lambda we found before
    # poissonDistribution = poisson.rvs(lambdaEstimate, size=len(values))
    # # plt.hist(values)
    # # plt.show()
    # # plt.hist(poissonDistribution)
    # # plt.show()
    # degreeSum = sum(values)
    # normalizedValues = [float(i)/degreeSum for i in values]
    # poissonSum = sum(poissonDistribution)
    # normalizedPoisson = [float(i)/poissonSum for i in values]

    # r2Score = r2_score(values,poissonDistribution)
    # print(f'{r2Score=}')
    # chiquareScore = chisquare(normalizedValues, f_exp=normalizedPoisson)
    # print(f'{chiquareScore=}')


def powerLawFit(G):
    pass
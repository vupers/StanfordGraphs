from random import randrange
import matplotlib.pyplot as plt
from scipy.stats import poisson, powerlaw
from sklearn.metrics import r2_score
def estimateValues(G):
    iterations = 100000
    counter = 0
    sumOfKv0 = 0
    sumOfKv1 = 0
    for _ in range(0,iterations):
        randomNode = randrange (0, G.number_of_nodes())
        numberOfNeighbors = sum(1 for _ in G.neighbors(randomNode))
        neighborRange = randrange(0, numberOfNeighbors)
        neighborIndex = -1
        neighborCounter = 0
        for neighbor in G.neighbors(randomNode):
            if neighborCounter == neighborRange:
                neighborIndex = neighbor
            neighborCounter +=1
        Kv0 = G.degree[randomNode]
        Kv1 = G.degree[neighborIndex]
        sumOfKv0 += Kv0
        sumOfKv1 += Kv1
        # print(f'v0 is node {randomNode}. v1 is node {neighborIndex}.')
        # print(f'Kv0 = {Kv0}, Kv1 = {Kv1}')
        if Kv1 > Kv0:
            counter = counter + 1
    expectedValueKv0 = sumOfKv0 / iterations
    expectedValueKv1 = sumOfKv1 / iterations
    print(f'E(Kv0) = {expectedValueKv0}, E(Kv1) = {expectedValueKv1}')
    print(f'out of {iterations} trials, Kv1 was larger than Kv0 {counter} times')

def poissonFit(G):
    #obtain lambda for poisson using MLE
    """ the estimator lambda_n is just the sample mean of the n observations in the sample.
    References: https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood
    https://en.wikipedia.org/wiki/Poisson_distribution#Parameter_estimation"""
    testRange= 100_000
    degreeSum = 0
    degrees = []
    for _ in range(0,testRange):
        randomNode = randrange (0, G.number_of_nodes())
        degreeSum += G.degree[randomNode]
        degrees.append(G.degree[randomNode])
    
    lambdaEstimate = degreeSum / testRange
    print(f'Using MLE, we obtain a value of {lambdaEstimate = }')
    poissonDistribution = poisson.rvs(lambdaEstimate, size=testRange)
    # plt.hist(poissonDistribution)
    # plt.hist(degrees)

    r2Score = r2_score(degrees,poissonDistribution)
    print(f'{r2Score=}')

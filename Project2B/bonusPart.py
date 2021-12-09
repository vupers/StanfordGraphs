from random import randrange
import matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import networkx as nx
import statsmodels.api as sm
from math import factorial,exp
import powerlaw
import numpy as np
def estimateValues(G, results_dir):
    print ('Estimating values...\n')

    iterations = G.number_of_nodes() * 10
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

def fit_function(k, lamb):
    return poisson.pmf(k, lamb)

def poissonFit2(G):
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    
    # fit with curve_fit
    parameters, cov_matrix = curve_fit(fit_function, degrees,degree_freq)
    print(parameters)
    values = []
    for value in degrees:
        values.append(fit_function(value,parameters[0]))
    
    return parameters[0]
    

def poissonFunc(x, lambdavalue):
    return exp(-lambdavalue) * pow(lambdavalue,x) / factorial(x)

def obtainLambda(G):
    testRange= G.number_of_nodes() * 10
    # degreeSum = 0
    degrees = []
    for _ in range(0,testRange):
        randomNode = randrange (0, G.number_of_nodes())
        if not G.has_node(randomNode):
            #print('[POISSON]randomNode is broken :((')
            continue
        # degreeSum += G.degree[randomNode]
        degrees.append(G.degree[randomNode])

    degreeSum = sum(degrees)
    lambdaEstimate = degreeSum / len(degrees)
    print(f'Using MLE, we obtain a value of {lambdaEstimate = }.')
    return lambdaEstimate

def func(G,results_dir, fname):
    lambdaEstimate = obtainLambda(G)
    degree_freq = nx.degree_histogram(G)
    degrees = np.arange(len(degree_freq))
    degree_normalized = []
    num = G.number_of_nodes()
    #print(G.degree)
    for x in degree_freq:
        degree_normalized.append(x/num)
    
    poissonPMF = [poisson.pmf(i,lambdaEstimate) for i in degrees]

    cut=100
    # degrees = degrees[cut:]
    # degree_normalized = degree_normalized[cut:]
    # poissonPMF = poissonPMF[cut:]
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Poisson comparison')
    ax = plt.subplot(111)
    ax.bar(degrees-0.2, degree_normalized, width=0.4, color='b', align='center', label='Empirical Distribution')
    ax.bar(degrees + 0.2, poissonPMF, width=0.4, color='y', align='center',label = 'Poisson Distribution for p->1')
    plt.legend(loc='upper left')
    plt.show()
    r2Score = r2_score(poissonPMF,degree_normalized)
    print(f'{r2Score=}')
   

    

def poissonFit(G,results_dir, fname):
    #obtain lambda for poisson using MLE
    """ the estimator lambda_n is just the sample mean of the n observations in the sample.
    References: https://en.wikipedia    .org/wiki/Poisson_distribution#Parameter_estimation
    https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood"""
    print ('Determining value of lambda...\n')
    # lambdaEstimate = obtainLambda(G)
    #generate the list that contains all the degrees of the graph
    values = []
    for node in list(G.nodes):
        values.append(G.degree[node])
    # poissonDistribution = poisson.rvs(lambdaEstimate, size=len(values))
    # print(poissonDistribution)
    degreeSum = sum(values)
    normalizedValues = [float(i)/degreeSum for i in values]
    
    # poissonSum = sum(poissonDistribution)
    # normalizedPoisson = [float(i)/poissonSum for i in poissonDistribution]

    
    lambdaOptimized = poissonFit2(G)
    optimizedPoissonDistribution = poisson.rvs(lambdaOptimized, size=len(values))
    optimizedPoissonSum = sum(optimizedPoissonDistribution)
    normalizedOptimizedPoisson = [float(i)/optimizedPoissonSum for i in optimizedPoissonDistribution]
    # optimizedPoissonSum = sum(optimizedPoissonDistribution)
    # normalizedOptimizedPoisson = [float(i)/poissonSum for i in optimizedPoissonDistribution]
    # anotherList =[]
    # for val in values:
    #     if(val < 50):
    #         anotherList.append(val)
    # plt.title(f"Comparison for {fname[:-4]}")
    # plt.xlabel("Degree")
    # plt.ylabel("Number of nodes")
    # # plt.xscale('log')
    # plt.hist([anotherList,poissonDistribution], bins=50,label=['Node Values','Poisson Distribution 1'])
    # #plt.hist(poissonDistribution, alpha=0.5, label='Poisson Distribution')
    # plt.legend(loc='upper right')
    # plt.show()

    # plt.title(f"Comparison for {fname[:-4]}")
    # plt.xlabel("Degree")
    # plt.ylabel("Number of nodes")
    # #plt.xscale('log')
    # plt.hist([anotherList,optimizedPoissonDistribution],bins=50, label=['Node Values','Poisson Distribution 2'])
    # #plt.hist(poissonDistribution, alpha=0.5, label='Poisson Distribution')
    # plt.legend(loc='upper right')
    # plt.show()

    # plt.xlabel("Degree")
    # plt.ylabel("Number of nodes")
    # plt.hist([anotherList,poissonDistribution,optimizedPoissonDistribution],bins=50, label=['Node Values','Poisson 1', 'Poisson 2'])
    # #plt.hist(poissonDistribution, alpha=0.5, label='Poisson Distribution')
    # plt.legend(loc='upper right')
    # plt.show()
    
    # r2Score = r2_score(normalizedValues,normalizedPoisson)
    # print(f'{r2Score=}')
    r2Score2 = r2_score(normalizedValues,normalizedOptimizedPoisson)
    print(f'Comparing optimized poisson. {r2Score2=}')

def powerLawFit(G):
    data = []
    for node in list(G.nodes):
        data.append(G.degree[node])
    #fit = powerlaw.Fit(np.array(data) + 1, discrete=True)
    fit = powerlaw.Fit(np.array(data) + 1, xmin = 5, discrete=True)
    fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit tail distribution')
    #fit2.power_law.plot_pdf( color= 'y',linestyle='--',label='fit tail distribution, fixed xmin')
    fit.plot_pdf( color= 'r', label = 'degree distribution')


    plt.legend(loc='upper right')
    plt.title ("Comparison between degree distribution and power law")
    plt.xlabel("Degree (log)")
    plt.ylabel("Number of nodes (log)")
    # figPDF = powerlaw.plot_pdf(data, color='b')
    # powerlaw.plot_pdf(data, linear_bins=True, color='r', ax=figPDF)
    # ####
    # figPDF.set_ylabel("Probability")
    # figPDF.set_xlabel(r"Degree")
    plt.show()
    #savefig(figname+'.tiff', bbox_inches='tight', dpi=300)
    print(fit.power_law.alpha)
    print(fit.power_law.xmin)
    R1, p1 = fit.distribution_compare('power_law', 'lognormal')
    print(f"loglikelihood = {R1}")
    int (f"significance of R = {p1}")
    # print("\nfor fixed xmin = 1:")
    # R2, p2 = fit2.distribution_compare('power_law', 'lognormal')
    # print(f"loglikelihood = {R2}")
    # print (f"significance of R = {p2}")
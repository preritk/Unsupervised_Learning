#K_Means Cluster Algorithm
#Unsupervised algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
def costfunc(set1,set2,set3,c1,c2,c3):
    J = 0.0
    a = len(set1)
    b = len(set2)
    c = len(set3)
    for i in range(a):
        J += math.pow(distance(set1[i],c1),2)
    for i in range(b):
        J += math.pow(distance(set2[i], c2),2)   ##Cost is calculated from all the K sets .
    for i in range(c):
        J += math.pow(distance(set3[i], c3),2)
    return J/150


def distance(instance,centroid):
    dist = 0.0
    #print(centroid)
    for i in range(len(instance)):
        dist = dist + math.pow(instance[i] - centroid[i] , 2)  #Eucledian distance formula
    return math.sqrt(dist)

def InitializeMean(K,X):
    ##Randomly choose any 3 datasets manually or use random function to generate random values of index.
    ##Here we will generate random function .
    i1 = random.randint(1,100)
    i2 = random.randint(50,149)   #You can specify some other range also .
    i3 = random.randint(1,149)
    return X[i1],X[i2],X[i3]

def main():
    X = pd.read_csv('iris.csv', header = None , usecols = [0,1,2,3]).values   ##Getting values of X(i)
    Y = pd.read_csv('iris.csv', header = None,usecols = [4]).values           ##Y values which are just for matching values at last
    for i in range(len(X)):
        for j in range(4):
            X[i][j] = float(X[i][j])                                          ##converting object values to float

    ###Make possible combinations of 2 features and plot graphs to get some intuition .
    ###This will help us know how many clusters can be there by visualisation.
    for i in range(len(X)):
        plt.scatter(X[i][0],X[i][1],marker = 'x',color = 'b')
    plt.show()
    for i in range(len(X)):
        plt.scatter(X[i][2], X[i][3], marker='x', color='b')
    plt.show()
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][2], marker='x', color='b')
    plt.show()
    for i in range(len(X)):
        plt.scatter(X[i][1], X[i][3], marker='x', color='b')
    plt.show()
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][3], marker='x', color='b')
    plt.show()
    for i in range(len(X)):
        plt.scatter(X[i][1], X[i][2], marker='x', color='b')
    plt.show()
    ##From the above 6 plots we get to know that there are minimum 2 clusters and we have to find the correct number of clusters.
    ##We can find it out by using elbow method or run the program using different values of K and that K for which cost
    ## is minimum is the optimal number
    ##of custer.
    ##Here we have prior knowledge of types of iris so we intuitively take K = 3 .
    K = 3
    m = len(X)
    iterator, cost = [], []
    for k in range(50):
        c1, c2, c3 = InitializeMean(K, X)
        temp1, temp2, temp3 = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
        dist1, dist2, dist3 = 0.0, 0.0, 0.0
        set1, set2, set3 = [], [], []
        A = 0
        while(A != 1000):
            set1,set2,set3 = [],[],[]
            for i in range(m):
                dist1 = distance(X[i],c1)
                dist2 = distance(X[i],c2)  #finding distances from all the centroids
                dist3 = distance(X[i],c3)
                #Now find minimum of all these distances .
                min = dist1
                if(dist2<min):
                    min = dist2
                if(dist3<min):
                    min = dist3

                if(min==dist1):
                    set1.append(X[i])
                elif(min==dist2):
                    set2.append(X[i])      #finding closets of all centroids
                else:
                    set3.append(X[i])
            #Now we have to update centroids .
            #Updating centroids as follows::
            #Take mean of all the points in the closets
            c1 = np.mean(set1,axis = 0)
            c2 = np.mean(set2,axis = 0)
            c3 = np.mean(set3,axis = 0)
            A += 1
        J = costfunc(set1,set2,set3,c1,c2,c3)
        iterator.append(k)
        cost.append(J)
    plt.plot(iterator,cost,'-',color = 'b')
    plt.show()
    print("WORK FINISHED !!!")
    ##NOW YOU WILL GWT A PLOT OF COST . CHOOSE THAT POINT WHERE COST IS MINIMUM .
    ##SO THIS DIVIDED THE DATASET INTO K CLUSTERS.
main()

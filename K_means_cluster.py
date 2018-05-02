#K_Means Cluster Algorithm
#Unsupervised algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
def calc_closets(c1,c2,c3,X):   ##This function defines closets of each point
    return 0

def distance(instance,centroid):
    dist = 0.0
    for i in range(len(instance)):
        dist = dist + math.pow(instance[i] - centroid[i],2)
    return math.sqrt(dist)

def InitializeMean(K,X):
    ##Randomly choose any 3 datasets manually or use random function to generate random values of index.
    ##Here we will generate random function .
    c1,c2,c3 =[],[],[]
    i1 = random.randint(1,50)
    i2 = random.randint(51,100)
    i3 = random.randint(101,150)
    return c1,c2,c3

def main():
    X = pd.read_csv('iris.csv', header = None , usecols = [0,1,2,3]).values   ##Getting values of X(i)
    Y = pd.read_csv('iris.csv', header = None,usecols = [4]).values           ##Y values which are just for matching values at last
    for i in range(len(X)):
        for j in range(4):
            X[i][j] = float(X[i][j])                                          ##converting object values to float

    ###Make possible combinations of 2 features and plot graphs to get some intuition .
    ###This will help us know how many clusters can be there by visualisation.
    """for i in range(len(X)):
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
    plt.show()"""
    ##From the above 6 plots we get to know that there are minimum 2 clusters and we have to find the correct number of clusters.
    ##We can find it out by using elbow method or run the program using different values of K and that K for which cost
    ## is minimum is the optimal number
    ##of custer.
    ##Here we have prior knowledge of types of iris so we intuitively take K = 3 .
    K = 3
    m = len(X)
    i1,i2,i3 = InitializeMean(K,X)      
    temp1,temp2,temp3 =0,0,0
    dist1,dist2,dist3 = 0.0,0.0,0.0
    set1,set2,set3 = [],[],[]
    while(1):
        max = 0.0
        set1,set2,set3 = [],[],[]
        for i in range(m):                  ##This loop is to find the distance of each point with the cluster centroid
            dist1 = distance(X[i],X[i1])    ##Distance from cluster centroid1
            dist2 = distance(X[i],X[i2])    ##Distance from cluster centroid2
            dist3 = distance(X[i],X[i2])    ##Distance from cluster centroid3
            if(dist1>max):
                max = dist1
            if(dist2>max):
                max = dist2                 ##Finding max of all three distances
            if(dist3>max):
                max = dist3

            if(max==dist1):
                set1.append(i)
            elif(max==dist2):
                set2.append(i)              ##Finding the closure set of each centroid .
            else:
                set3.append(i)

        i1 = int(np.mean(set1))                  ##centroid of first cluster
        i2 = int(np.mean(set2))                  ##centroid of second cluster
        i3 = int(np.mean(set3))                  ##centroid of third cluster
        if(i1==temp1 and i2==temp2 and i3 == temp3):  ##checking whether centroids moved or not
            break
        if(i1 != temp1):
            temp1 = i1
        if(i2 != temp2):
            temp2 = i2
        if(i3 != temp3):
            temp3 = i3




main()

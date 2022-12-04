#-------------------------------------------------------------------------
# AUTHOR: Andy Lam
# FILENAME: CS4210Assignment5
# SPECIFICATION: To run k-means multiple times and check which k value maximizes the
# Silhouette coefficient. You also need to plot the values of k and their
# corresponding Silhouette coefficients so that we can visualize and confirm the
# best k value found. Next, you will calculate and print the Homogeneity score
# (the formula of this evaluation metric is provided in the template) of this
# best k clustering task by using the testing_data.csv, which is a file that
# includes ground truth data.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library
highestMax = 0.0
silhouette_coefficient = []
#assign your training data to X_training feature matrix
X_training = df
for k in range (2,21):
#run kmeans testing different k values from 2 until 20 clusters
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)
     #--> add your Python code

     currentMax = silhouette_score(X_training, kmeans.labels_)
     silhouette_coefficient.append(currentMax)
     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     if currentMax > highestMax:
          highestMax = currentMax
          kMax = k
     #--> add your Python code here

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(silhouette_coefficient)
plt.show()
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
abc = pd.read_csv('testing_data.csv', sep=',', header=None)
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(abc.values).reshape(1,abc.shape[0])[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here

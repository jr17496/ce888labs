import pandas
import numpy as np
import pandas as pd

#from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


import matplotlib.pyplot as plt


import seaborn as sns

df = pd.read_csv("hUSCensus1990raw50K.csv.bz2",compression = "bz2")

print(df.head())
print(list(df))

plot_df = pd.DataFrame()
plot_df["AGE"] = df[["AGE"]].copy()
plot_df["INCOME"] = df[["INCOME" + str(i) for i in range(1,8)]].sum(axis = 1)


sns_plot = sns.pairplot(plot_df)
sns_plot.savefig('Initial.png')



from sklearn import metrics
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df_values = plot_df.sample(n= 1000).values


X_db = sc.fit_transform(df_values)

labels = []
min_score = 9999999

for i in range (2,10):
    n_clusters =i
    labels.append(KMeans(n_clusters=n_clusters).fit_predict(X_db))
for i in range(0,len(labels)):
    number_of_clusters = i +2
    score = metrics.silhouette_score(X_db, labels[i])
    print('for %d clusters the silhouette score is %f'%(number_of_clusters,score))
    if score<min_score:
        min_score=score
        min_clusters = number_of_clusters

print('The lowest silhouette score was achived by %d clusters'%min_clusters)


min_score_agglomerative = 999999
labels_agglomerative = []

for i in range (2,10):
    n_clusters =i
    labels_agglomerative.append(AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_db))
for i in range(0,len(labels_agglomerative)):
    number_of_clusters_agglomerative = i +2
    score = metrics.silhouette_score(X_db, labels_agglomerative[i])
    print('for %d clusters the silhouette score is %f'%(number_of_clusters_agglomerative,score))
    if score<min_score_agglomerative:
        min_score_agglomerative=score
        min_clusters_agglomerative = number_of_clusters_agglomerative

print('The lowest silhouette score was achived by %d clusters'%min_clusters_agglomerative)








#df_demo = pd.DataFrame()
#df_demo["AGE"] = df[["AGE"]].copy()
#df_demo["INCOME"] = df[["INCOME" + str(i) for i in range(1,8)]].sum(axis = 1)

#df_demo["YEARSCH"] = df[["YEARSCH"]].copy()
#df_demo["ENGLISH"] = df[["ENGLISH"]].copy()
#df_demo["FERTIL"] = df[["FERTIL"]].copy()
#df_demo["YRSSERV"] = df[["YRSSERV"]].copy()



#df_demo = pd.get_dummies(df_demo, columns = ["ENGLISH", "FERTIL" ])



#X = df_demo.values[np.random.choice(df_demo.values.shape[0], 10000)]

#from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_db = sc.fit_transform(X)


#n_clusters = 3

#labels = KMeans(n_clusters = n_clusters).fit_predict(X_db)




#print('Number of clusters: %d' % n_clusters)

#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X_db, labels))

#unique_labels = set(labels)
#colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#for k, col in zip(unique_labels, colors):
#    if k == -1:
        # Black used for noise.
#        col = 'k'

#    class_member_mask = (labels == k)

#    xy = X[class_member_mask]
#    plt.scatter(xy[:, 0], xy[:, 1],  c = col, edgecolor='k')

#    xy = X[class_member_mask]
 #   plt.scatter(xy[:, 0], xy[:, 1],  c = col, edgecolor='k')

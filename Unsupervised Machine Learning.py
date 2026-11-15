# # K Means Clustering:
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import load_iris

# #1 LOADINF DATA
# iris = load_iris()
# X = iris.data[:, :2] #taking only the first 2 features sepal length and width
# print(X)

# #2 Model Prep and Fitting:
# #HWe know there are  types of flower in the dataset, so K=3 is taken
# kmeans = KMeans(n_clusters=3,random_state=42)
# kmeans.fit(X)

# #3 Prediction(which flower goes to which cluster)
# y_kmeans = kmeans.predict(X)
# centers = kmeans.cluster_centers_ #Gives centroids or center points

# #4 Visualizatin of results:
# plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis') #Data Points
# plt.scatter(centers[:, 0], centers[:,1], c='red', s=200, marker='X')#Centroid
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.title('Simple K-means Clustering on Iris Data')
# plt.show()


#HIERARCHIAL CLUSTERING
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as sch
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets import make_blobs

# #1 DUMMY DATA PREP(50 POINTS, 3 GROUPS)
# X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6, random_state=42)
# print(X, _)
# #2 DENDOGRAM PREP(TELLS US HOW MANY CLSUTER TO MAKE)
# # plt.figure(figsize=(10,5))
# # dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# # plt.title('Dendogram(Tree Diagram)')
# # plt.xlabel('Data Points')
# # plt.ylabel('Euclidean Distance')
# # plt.show

# plt.figure(figsize=(10, 5))

# Z = sch.linkage(X, method='ward')
# print("Linkage complete")

# dendrogram = sch.dendrogram(Z)
# print("Dendrogram complete")

# plt.title("Dendrogram")
# plt.xlabel("Data Points")
# plt.ylabel("Euclidean Distance")

# plt.show(block=True)

# #MODEL TRAINING(3 clsuters sicne the dendogram gives 3)
# model= AgglomerativeClustering(n_clusters=3, linkage='ward')
# Y_predict = model.fit_predict(X)

# #4 Results Visualization
# plt.scatter(X[:, 0], X[:,1], c=Y_predict, cmap='rainbow', s=50)
# plt.title('Hierarchia Clsutering Results')
# plt.show()


#DENSITY BASED CLUSTERING ALGORITHM
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

#GENERATION OF SHAPES
X, _ = make_moons(n_samples=250, noise=0.05, random_state=42)

#2 dbscan model
# EPS: radius for searching neighbours
#min_samles: min no of points needed to make a cluster
dbscan = DBSCAN(eps=0.2, min_samples=5)

#3 Model fitting and cluster prediction
Y_pred = dbscan.fit_predict(X)


#4 RESULT VISUALIZATION:
#Note: CLUSTER ID-1, means 'Noise' is outlier
plt.figure(figsize=(8,5))
plt.scatter(X[:, 0], X[:, 1], c=Y_pred, cmap='plasma', s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

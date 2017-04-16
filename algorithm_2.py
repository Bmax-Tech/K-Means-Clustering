import numpy
import pandas
import matplotlib.pyplot as mp_plot
import matplotlib.image as mp_img
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# read data set
data = pandas.read_csv("data/data_set_3.csv")
# remove unnecessary columns
data.__delitem__('Channel')
data.__delitem__('Region')

print("Dataset has {} rows, {} columns".format(*data.shape))
print(data.head())

pca = PCA(n_components=6)
pca.fit(data)

# display the components and variance of the data
print(pandas.DataFrame(pca.components_, columns=list(data.columns)))
print("-----------------------------------------------------------")
print("Variance Ratio of Individual Components")
print("-----------------------------------------------------------")
print(pca.explained_variance_ratio_)

mp_plot.figure(figsize=(11, 5))
mp_plot.plot(numpy.arange(1, 7), numpy.cumsum(pca.explained_variance_ratio_))
mp_plot.xlabel("number of PCA components")
mp_plot.ylabel("cumulative variance explained by each dimension")
mp_plot.title("Variance Trend by PCA Dimension")
mp_plot.savefig('images/app_2/variance.png')

# centring the data
centered = data.copy()
centered -= centered.mean()
ica = FastICA(n_components=6)
ica.fit(centered)

# plotting heat map for better visualisation of matrix
mp_plot.figure(figsize=(11, 5))
sns.heatmap(pandas.DataFrame(ica.components_, columns=list(data.columns)), annot=True)
mp_plot.savefig('images/app_2/heat_map.png')

# ----------------------------------------------------------------------------------------------------------------------
# Apply K-Means Clustering
# ----------------------------------------------------------------------------------------------------------------------

# 1. Reduce the data to 2 dimensions using PCA to capture variance
reduced_data = PCA(n_components=2).fit_transform(centered)
print("-----------------------------------------------------------")
print("PCA Applied dimensions")
print("-----------------------------------------------------------")
print(reduced_data[:10])  # print up to 10 elements

# 2. clusters
clusters = GaussianMixture(n_components=4).fit(reduced_data)
print(clusters)

# 3. Plot the decision boundary by building a mesh grid to populate a graph
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
hx = (x_max - x_min) / 1000.
hy = (y_max - y_min) / 1000.
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, hx), numpy.arange(y_min, y_max, hy))

# 4. Obtain labels for each points in the mesh. Use last trained model
Z = clusters.predict(numpy.c_[xx.ravel(), yy.ravel()])

# 5. centroids
centroids = clusters.means_
print("\nCentroids")
print(centroids)

# 6. put the results into a color plot
Z = Z.reshape(xx.shape)
mp_plot.figure(1)
mp_plot.clf()
mp_plot.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=mp_plot.cm.Paired,
               aspect='auto', origin='lower')
mp_plot.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
mp_plot.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
mp_plot.title('Clustering on the wholesale grocery dataset\nCentroids are marked with white cross')
mp_plot.xlim(x_min, x_max)
mp_plot.ylim(y_min, y_max)
mp_plot.xticks(())
mp_plot.yticks(())
mp_plot.savefig('images/app_2/clusters_1.png')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# Plot Heat map for given data
def plot_heat_map(percentiles_data, ax=None, figsize=(10, 5), title="Heatmap", index=0):
    fig, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)
    ax.set_title(title)
    ax.imshow(percentiles_data, cmap=plt.cm.Greys, interpolation='nearest')
    ax.set_xticks(np.arange(len(percentiles_data.columns.values)))
    ax.set_xticklabels(percentiles_data.columns.values, rotation=90)
    ax.set_yticks(np.arange(len(percentiles_data.index.values)))
    ax.set_yticklabels(percentiles_data.index.values)
    for i, feature in enumerate(percentiles_data):
        for j, _ in enumerate(percentiles_data[feature]):
            ax.text(i, j, "{:0.2f}".format(percentiles_data.iloc[j, i]), verticalalignment='center',
                    horizontalalignment='center', color=plt.cm.Reds(1 - percentiles_data.iloc[j, i]), fontweight='bold')
    fig.savefig('images/app_3/percentiles_%s.png' % index)


# Predict the Feature
def predict_feature(feature, data):
    new_data = data.drop([feature], axis=1)

    # Split the data into training and testing sets using the given feature as the target
    x_train, x_test, y_train, y_test = train_test_split(new_data, data[feature], test_size=0.25, random_state=42)
    # Create a decision tree regression and fit it to the training set
    regression = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)

    # Retort the score of the prediction using the testing set
    score = regression.score(x_test, y_test)

    print("The Score for feature {:16} id {:+.5f}".format(feature, score))


# Score Clustering
def score_clustering(data, num_custers, pca_s):
    preds, _, _ = cluster(data, num_custers, pca_s)
    score = silhouette_score(data, preds)
    return score


# Make clusters
def cluster(data, num_clusters, pca_s):
    clusterer = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=42).fit(data)
    preds = clusterer.predict(data)
    centers = clusterer.means_
    sample_preds = clusterer.predict(pca_s)
    return preds, centers, sample_preds

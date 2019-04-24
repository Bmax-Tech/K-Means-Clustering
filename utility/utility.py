import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def cluster_results(reduced_data, preds, centers, pca_samples):
    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis=1)
    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))
    # Color map
    cmap = cm.get_cmap('gist_rainbow')
    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2',
                     color=cmap((i) * 1.0 / (len(centers) - 1)), label='Cluster %i' % (i), s=30)

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x=c[0], y=c[1], color='red', edgecolors='black', alpha=1, linewidth=2, marker='o', s=200)
        ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100)

    # Plot transformed sample points
    ax.scatter(x=pca_samples[:, 0], y=pca_samples[:, 1], s=150, linewidth=2, color='black', marker='x')
    # Set plot title
    ax.set_title(
        "PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by X")
    # fig.savefig('images/app_3/plot_map.png')


def channel_results(reduced_data, outliers, pca_samples):
    # Check that the dataset is loadable
    try:
        full_data = pd.read_csv("/workspace/K-Means-Clustering/data/data_set_3.csv")
    except:
        print
        "Dataset could not be loaded. Is the file missing?"
        return False

    # Create the Channel DataFrame
    channel = pd.DataFrame(full_data['Channel'], columns=['Channel'])
    channel = channel.drop(channel.index[outliers]).reset_index(drop=True)
    labeled = pd.concat([reduced_data, channel], axis=1)
    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))
    # Color map
    cmap = cm.get_cmap('gist_rainbow')
    # Color the points based on assigned Channel
    labels = ['Hotel/Restaurant/Cafe', 'Retailer']
    grouped = labeled.groupby('Channel')
    for i, channel in grouped:
        channel.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', color=cmap((i - 1) * 1.0 / 2),
                     label=labels[i - 1], s=30)

    # Plot transformed sample points
    for i, sample in enumerate(pca_samples):
        ax.scatter(x=sample[0], y=sample[1], s=200, linewidth=3, color='black', marker='o', facecolors='none')
        ax.scatter(x=sample[0] + 0.25, y=sample[1] + 0.3, marker='$%d$' % (i), alpha=1, s=125)

    # Set plot title
    ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled")
    # fig.savefig('images/app_3/plot_map_2.png')

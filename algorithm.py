import numpy as np
import pandas as pd
import utility.utility as utility
import itertools as itr
from sklearn.mixture import GaussianMixture

from IPython.display import display
from sklearn.decomposition import PCA


def cluster_results(cluster_type):
    try:
        data = pd.read_csv('data/data_set_3.csv')
        data.drop(['Region', 'Channel'], axis=1, inplace=True)
        print("Wholesale customer dataset has {} samples with {} features each.".format(*data.shape))
        # Display a description of the dataset
        display(data.describe())

        # selected few data samples from dataset to further analysis
        # 43: Very low "Fresh" and very high "Grocery"
        # 12: Very low "Frozen" and very high "Fresh"
        # 39: Very high "Frozen" and very low "Detergens_Paper"
        indices = [43, 12, 39]

        # Create Data frames for the selected samples
        samples = pd.DataFrame(data.loc[indices], columns=data.columns).reset_index(drop=True)
        print("Chosen samples of wholesale customers dataset:")
        display(samples)

        # Scale the data using the natural logarithm
        log_data = np.log(data)
        # Scale the sample data using the natural logarithm
        log_samples = np.log(samples)
        # Select the indices for data points you wish to remove
        outliers_lst = []

        # For each feature find the data points with extreme high or low values
        for feature in log_data.columns:
            # Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(log_data.loc[:, feature], 25)
            # Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(log_data.loc[:, feature], 75)
            # Use the inter quartile range to calculate an outlier step (1.5 times the inter quartile range)
            step = 1.5 * (Q3 - Q1)
            # Display the outliers
            print("Data points considered outliers for the feature '{}':".format(feature))
            # find any points outside of Q1 - step and Q3 + step (sign ~ means not)
            outliers_rows = log_data.loc[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step)), :]
            # display(outliers_rows)
            outliers_lst.append(list(outliers_rows.index))

        outliers = list(itr.chain.from_iterable(outliers_lst))
        # List of unique outliers
        unique_outliers = list(set(outliers))
        # List of duplicate outliers
        dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
        print('Outliers list:\n', unique_outliers)
        print('Length of outliers list:\n', len(unique_outliers))
        print('Duplicate list:\n', dup_outliers)
        print('Length of duplicates list:\n', len(dup_outliers))
        # Remove duplicate outliers
        # Only 5 specified
        good_data = log_data.drop(log_data.index[dup_outliers]).reset_index(drop=True)
        # Original Data
        print('Original shape of data:\n', data.shape)
        # Processed Data
        print('New shape of data:\n', good_data.shape)

        # Apply PCA by fitting the good data with only two dimensions
        # Instantiate
        pca = PCA(n_components=2)
        pca.fit(good_data)
        # Transform the good data using the PCA fit above
        reduced_data = pca.transform(good_data)
        # Transform the sample log-data using the PCA fit above
        pca_samples = pca.transform(log_samples)
        # Create a DataFrame for the reduced data
        reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
        # Display sample log-data after applying PCA transformation in two dimensions
        display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2']))

        # Make clusters
        cluster = GaussianMixture(n_components=cluster_type['id']).fit(reduced_data)
        predictions = cluster.predict(reduced_data)
        centers = cluster.means_
        # Display the results of the clustering from implementation
        utility.cluster_results(reduced_data, predictions, centers, pca_samples)

        # Display the clustering results based on 'Channel' data
        # utility.channel_results(reduced_data, dup_outliers, pca_samples)

        predictions = pd.DataFrame(predictions, columns=['Cluster'])
        plot_data = pd.concat([predictions, reduced_data], axis=1)

        return plot_data, centers
    except:
        print("Data set could not be loaded.")
        return None

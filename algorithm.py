import numpy as np
import pandas as pd
import utility.utility as utility

from IPython.display import display  # allows the use of display for Data frames
from sklearn.decomposition import PCA
from collections import Counter


def cluster_results(cluster_type):
    try:
        data = pd.read_csv('data/data_set_3.csv')
        data.drop(['Region', 'Channel'], axis=1, inplace=True)
        print("Wholesale customer dataset has {} samples with {} features each.".format(*data.shape))

        display(data.describe())

        # selected few data samples from dataset to further analysis
        indices = [40, 80, 150]

        # Create Dataframes for the selected samples
        samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
        print("Selected samples of wholesale customer dataset")
        display(samples)

        # plot heat map
        samples_percentiles = data.rank(pct=True).loc[indices]
        utility.plot_heat_map(samples_percentiles, figsize=(7, 5), title="Percentiles", index=1)

        # Predict features
        for feature in data.columns.values:
            utility.predict_feature(feature, data)

        # Scale the data using the natural logarithm
        log_data = np.log(data)
        # Scale the sample data using the natural logarithm
        log_samples = pd.DataFrame(log_data.loc[indices], columns=data.keys()).reset_index(drop=True)
        display(log_samples)

        print("\nOutliers\n")
        outliers_counter = Counter()
        outliers_scores = None
        # for each feature find the data points with extreme high or low values
        for feature in log_data.keys():
            # Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(log_data[feature], 25)
            # Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(log_data[feature], 75)
            # Use the inter quartile range to calculate an outlier step (1.5 times the inter quartile range)
            steps = 1.5 * (Q3 - Q1)
            zeros = np.zeros(len(log_data[feature]))
            above = log_data[feature].values - Q3 - steps
            below = log_data[feature].values - Q1 + steps
            current_outliers_scores = np.array(np.maximum(zeros, above) - np.minimum(zeros, below)).reshape([-1, 1])
            outliers_scores = current_outliers_scores if outliers_scores is None else np.hstack(
                [outliers_scores, current_outliers_scores])

            # Display the outliers
            print("Data points considered outliers for the feature '{}' : ".format(feature))
            current_outliers = log_data[~((log_data[feature] >= Q1 - steps) & (log_data[feature] <= Q3 + steps))]
            display(current_outliers)
            outliers_counter.update(current_outliers.index.values)

        # Select the indices for data points you wish to remove
        min_outliers_count = 2
        outliers = [x[0] for x in outliers_counter.items() if x[1] >= min_outliers_count]
        print("Data points considered outlier for more than 1 feature: {}".format(outliers))

        # Remove the outliers, if any were specified
        good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

        # Outlier heapmap scores
        utility.plot_heat_map(pd.DataFrame(data=outliers_scores[outliers], index=outliers, columns=log_data.columns),
                              title="Five outliers with their scores", index=2)

        # Apply PCA by filtering the good data with the same number of dimensions as features
        # pca = PCA(n_components=len(good_data.columns)).fit(good_data)
        # Transform the sample log-data using the PCA fit above
        # pca_samples = pca.transform(log_samples)

        # Apply PCA by filtering the good data with only two dimensions
        pca = PCA(n_components=2).fit(good_data)
        # Transform the good data using the PCA fit above
        reduced_data = pca.transform(good_data)
        # Transform the sample log-data using the PCA fit above
        # pca_samples = pca.transform(log_samples)
        pca_samples = pca.transform(log_data)
        # Create a Data frame for the reduced data
        reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
        # display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2']))
        data_points = pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2'])

        # Calculate scores for each clusters
        for num_clusters in range(2, 10):
            score = utility.score_clustering(reduced_data, num_clusters, pca_samples)
            print("num_clusters: {} - score: {}".format(num_clusters, score))

        preds, centers, sample_preds = utility.cluster(reduced_data, cluster_type['id'], pca_samples)

        # Implement Data Recovery

        # Inverse transform the centers
        log_centers = pca.inverse_transform(centers)
        # Exponentiate the centers
        true_centers = np.exp(log_centers)
        # Display the true centers
        segments = ['Segment {}'.format(i) for i in range(0, len(centers))]
        true_centers = pd.DataFrame(np.round(true_centers), columns=data.keys())
        true_centers.index = segments
        display(true_centers)

        # Display the predictions
        # for i, pred in enumerate(sample_preds):
        #     print("Sample points", i, "predicate to be in cluster", pred)

        return data_points.get_values(), sample_preds
    except:
        print("Data set could not be loaded.")
        return None

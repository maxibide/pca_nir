import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from scipy.signal import savgol_filter


class PCAObject:
    def __init__(self, df, components=3):
        """
        Class for performing Principal Component Analysis (PCA) and clustering analysis on the data.

        Parameters:
        - df (DataFrame): DataFrame containing the data.
        - components (int): Number of principal components to retain. Default is 3.
        """

        self.components = components
        self.df = df

        # Read data into an array (scans)
        feat = self.df.values.astype('float32')

        # Scale the data
        self.scaler = StandardScaler()
        self.scaler.fit(feat)
        scaled_feat = self.scaler.transform(feat)

        # Obtain principal components
        self.pca = PCA(n_components=self.components).fit(scaled_feat)
        self.pc = pd.DataFrame(self.pca.transform(scaled_feat), index=df.index)
        self.kmeans = None
        self.sil_score = None
        self.davies_bouldin = None
        self.calinski_harabasz = None
        self.rand_score = None

    def get_loadings(self):
        """
        Returns the PCA loadings.

        Returns:
        - loadings_df (DataFrame): DataFrame containing the PCA loadings.
        """

        loadings = self.pca.components_.T * \
            np.sqrt(self.pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings).transpose()
        loadings_df.columns = self.df.columns

        return loadings_df

    def determine_clusters(self):
        """
        Determines the optimal number of clusters using clustering methods.
        """

        # Elbow method
        inertia = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(self.pc)
            inertia.append(kmeans.inertia_)

        plt.plot(range(2, 11), inertia, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.show()

        # Silhouette criterion
        silhouette_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(self.pc)
            score = silhouette_score(self.pc, kmeans.labels_)
            silhouette_scores.append(score)

        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Criterion')
        plt.show()

    def cluster(self, n_clusters):
        """
        Clusters the data using the K-means algorithm.

        Parameters:
        - n_clusters (int): Number of clusters to create.
        """

        # Cluster the data using K-means
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        self.kmeans.fit(self.pc)
        clusters = self.kmeans.labels_

        # Calculate silhouette score
        self.sil_score = silhouette_score(self.pc, self.kmeans.labels_)

        # Davies-Bouldin Score
        self.davies_bouldin = davies_bouldin_score(
            self.pc, self.kmeans.labels_)

        # Calinski-Harabasz Score
        self.calinski_harabasz = calinski_harabasz_score(
            self.pc, self.kmeans.labels_)

        self.pc["cluster"] = clusters
        self.pc.set_index('cluster', append=True, inplace=True)

        # Adjusted Rand Score
        self.adj_rand_score = self._calculate_rand_score()

        print("Predicted cluster: ", clusters)
        print("Silhouette score: ", self.sil_score)
        print("Davies Bouldin Score: ", self.davies_bouldin)
        print("Calinski Harabasz Score: ", self.calinski_harabasz)
        print("Adjusted Rand Score: ", self.adj_rand_score)

    def _calculate_rand_score(self):
        """
        Calculates the adjusted Rand score.
        """

        cluster_index = self.pc.index.get_level_values('cluster')

        # Create a dictionary to store Rand adjusted scores
        rand_scores = {}

        # Iterate over index names in the DataFrame
        for index_name in self.pc.index.names:
            if index_name != 'cluster':  # Avoid comparing "cluster" index against itself
                other_index = self.pc.index.get_level_values(index_name)
                rand_score = adjusted_rand_score(cluster_index, other_index)
                rand_scores[index_name] = rand_score

        # Convert the dictionary of Rand adjusted scores to a DataFrame for visualization
        rand_scores_df = pd.DataFrame.from_dict(
            rand_scores, orient='index', columns=['Adjusted Rand Score'])

        return rand_scores_df

    def plot_pca(self, title, sizex=10, sizey=10, color_index=0, legend_index=0, legend=False, annotated=True, add_clusters=False, dimensions=2):

        plot_pca(self.pc, title, sizex, sizey, color_index, legend_index, legend, annotated,
                 add_clusters, dimensions, sil_score=self.sil_score, adj_rand_score=self.adj_rand_score)

    def predict_samples(self, df_val):
        """
        Predicts the cluster labels for a new set of data using the previously trained clustering model.

        Args:
            df_val (pandas.DataFrame): New set of data for which cluster labels are to be predicted.

        Returns:
            pandas.DataFrame: DataFrame with the cluster predictions and a "prediction" identifier for each sample.

        Raises:
            ValueError: If no clustering model has been trained previously.
        """
        if self.kmeans is None:
            raise ValueError(
                "No cluster model found. Run the cluster method first.")

        # Read data into an array (scans)
        np_val = df_val.values.astype('float32')

        # Scale the new data using the same scaler as the original data
        scaled_val = self.scaler.transform(np_val)

        # Transform the scaled new data to PCA space
        predict_pca = self.pca.transform(scaled_val)

        # Predict cluster labels for the new data
        y_pred = self.kmeans.predict(predict_pca)

        # Create a new DataFrame with the predictions
        df_pred = df_val.copy()
        df_pred['cluster'] = y_pred
        df_pred['type'] = 'prediction'

        # Set the cluster and type as the index
        df_pred.set_index(['cluster', 'type'], append=True, inplace=True)

        return df_pred


def plot_pca(df, title, sizex=10, sizey=10, color_index=0, legend_index=0, legend=False, annotated=True, add_clusters=False, dimensions=2, sil_score=None, adj_rand_score=None):
    """
    Plots the principal components.

    Parameters:
    - title (str): Title of the plot.
    - sizex (int): Size on the x-axis of the figure. Default is 10.
    - sizey (int): Size on the y-axis of the figure. Default is 10.
    - color_index (int): Index for color in the DataFrame. Default is 0.
    - legend_index (int): Index for legend in the DataFrame. Default is 0.
    - legend (bool): Boolean to include the legend box. Default is False.
    - annotated (bool): Boolean to include labels on the points. Default is True.
    - add_clusters (bool): Boolean to add cluster information on the plot. Default is False.
    - dimensions (int): Number of dimensions for the plot (2 or 3). Default is 2.
    """

    px = df.iloc[:, 0].tolist()
    py = df.iloc[:, 1].tolist()

    # List to store legend labels and colors for legend box
    legend_labels = []
    legend_colors = []

    # Store unique values of self.pc.index[i][color_index] in unique_values
    unique_values = np.unique([df.index[i][color_index]
                               for i in range(len(px))])

    # Create a dictionary mapping each unique value to a color
    color_map = {value: plt.cm.jet(i/len(unique_values))
                 for i, value in enumerate(unique_values)}

    if dimensions == 2:

        plt.figure(figsize=(sizex, sizey))

        for i in range(len(px)):
            plt.scatter(
                px[i], py[i], c=color_map[df.index[i][color_index]])

            if annotated:
                plt.annotate(df.index[i][legend_index], (px[i], py[i]),
                             textcoords="offset points", xytext=(0, 10), ha='center')

        plt.xlabel('PC1')
        plt.ylabel('PC2')

    elif dimensions == 3:

        pz = df.iloc[:, 2].tolist()

        fig = plt.figure(figsize=(sizex, sizey))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(px)):
            ax.scatter(px[i], py[i], pz[i],
                       c=color_map[self.pc.index[i][color_index]])

            if annotated:
                ax.text(px[i], py[i], pz[i], df.index[i][legend_index],
                        ha='center', fontsize=10)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    if legend:
        handles = [Patch(color=c, label=l)
                   for c, l in zip(legend_colors, legend_labels)]
        plt.legend(handles=handles, loc='upper right', markerscale=2, fontsize=10, title="Legend",
                   title_fontsize='13', facecolor='lightgrey', edgecolor='black',
                   fancybox=True, shadow=True)

    if add_clusters:
        if dimensions == 2:
            plt.text(0.95, 0.05, "Silhouette score: " + str(sil_score) + "\n" +
                     str(adj_rand_score), ha='right', va='bottom', transform=plt.gca().transAxes)
        elif dimensions == 3:
            ax.text2D(0.05, 0.95, "Silhouette score: " + str(sil_score) + "\n" +
                      str(adj_rand_score), ha='right', va='bottom', transform=ax.transAxes)

    plt.title(title)
    plt.show()

# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use

        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot

        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here

    # Loading the data and separating it by a comma 
    data_array = np.genfromtxt(datafile_name, delimiter=',')

    # Performing K-means clustering
    main_cluster_model = KMeans(n_clusters=K, n_init=10)
    main_cluster_model.fit(data_array)
    cluster_labels = main_cluster_model.predict(data_array)

    # Setting up the plot grid
    num_features = data_array.shape[1]
    fig, ax = plt.subplots(num_features, num_features, figsize=(12, 12))
    plt.set_cmap('viridis')

    #creating  Set colors for histograms
    hist_colors = plt.get_cmap('viridis', K).colors

    # Looping  through each feature pair
    feature1 = 0
    while feature1 < num_features:
        ax[feature1, 0].set_ylabel(feature_names[feature1])
        ax[0, feature1].set_xlabel(feature_names[feature1])
        ax[0, feature1].xaxis.set_label_position('top')

        feature2 = 0
        while feature2 < num_features:
            plot_data_x = data_array[:, feature1].copy()
            plot_data_y = data_array[:, feature2].copy()

            # Sorting  the data by cluster labels
            order_idx = np.argsort(cluster_labels)
            ordered_x = plot_data_x[order_idx]
            ordered_y = plot_data_y[order_idx]

            if feature1 != feature2:
                #making  Scatter plot for feature pairs
                ax[feature1, feature2].scatter(ordered_x, ordered_y, c=cluster_labels, s=50, marker='^', edgecolor='black', alpha=0.7)
            else:
                #making  Histogram for single feature 
                hist_idx = np.argsort(cluster_labels)
                hist_data = plot_data_x[hist_idx]
                hist_labels = cluster_labels[hist_idx]

                split_indices = np.unique(hist_labels, return_index=True)[1][1:]
                data_splits = np.split(hist_data, split_indices)

                k = 0
                while k < K:
                    ax[feature1, feature2].hist(data_splits[k], bins=20, color=hist_colors[k], edgecolor='black', alpha=0.7)
                    k += 1

            feature2 += 1
        feature1 += 1

    # Adding my username as the question asked in the begining and making it and writtig visualization cluster by k value 
    user_name = "(s2-kattel)"
    fig.suptitle(f'Visualisation of {K} clusters by {user_name}', fontsize=16, y=0.925)

    # Saveing the plot as myvisualization.jpg
    fig.savefig('myVisualisation.jpg')

    return fig, ax 

    # <==== insert your code above here

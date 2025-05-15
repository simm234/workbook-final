from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to  complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object

    ax: matplotlib.axes.Axes
        axis
    """

    # ====> insert your code below here


    # Defining the range of hidden layer widths (from 1 to 10 neurons)
    hidden_layer_widths = list(range(1, 11))

    # using array to store the number of successful runs (100% training accuracy) for each hidden layer width
    successful_runs = np.zeros(10)

    #  using matrix to store the number of epochs taken in each run [width_index, repetition_index]
    epoch_tracker = np.zeros((10, 10))


    for width_index, hidden_neurons in enumerate(hidden_layer_widths):
        # Running 10 experiments for each width with different random seeds
        for run_index in range(10):
            # Creating MLP with `hidden_neurons` in one hidden layer, fixed random seed
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_neurons,),
                max_iter=1000,
                alpha=1e-4,
                solver="sgd",
                learning_rate_init=0.1,
                random_state=run_index
            )

            # Training the model on XOR data
            model.fit(train_x, train_y)

            # Calculating training accuracy
            accuracy = 100 * model.score(train_x, train_y)


            if accuracy == 100:
                successful_runs[width_index] += 1
                epoch_tracker[width_index][run_index] = model.n_iter_

    # Computing average epochs for successful runs; assign 1000 if no success for that width
    avg_epochs = np.zeros(10)
    for i in range(10):
        if successful_runs[i] == 0:
            avg_epochs[i] = 1000
        else:
            avg_epochs[i] = np.mean(epoch_tracker[i][epoch_tracker[i] > 0])

    # Creating the side-by-side reliability and efficiency plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Reliability (success rate vs. hidden layer width)
    ax[0].plot(hidden_layer_widths, successful_runs / 10)
    ax[0].set_title("Reliability")
    ax[0].set_ylabel("Success Rate")
    ax[0].set_xlabel("Hidden Layer Width")

    # Right plot: Efficiency (mean epochs vs. hidden layer width)
    ax[1].plot(hidden_layer_widths, avg_epochs)
    ax[1].set_title("Efficiency")
    ax[1].set_ylabel("Mean Epochs")
    ax[1].set_xlabel("Hidden Layer Width")


    # <==== insert your code above here

    return fig, ax

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """ 

    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.

        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        self.data_x = np.genfromtxt(datafilename, delimiter=",")
        self.data_y = np.genfromtxt(labelfilename, delimiter=",").astype(int)
        # <==== insert your code above here

    def preprocess(self):
        """ Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if ther are more than 2 classes

           Remember to set random_state = 12345 if you use train_test_split()
        """

        # ====> insert your code below here
        # Spliting the dataset into training and testing sets using a stratified 70:30 split it ensures distribution in both side.

        # Stratified split (70:30)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, stratify=self.data_y, random_state=12345
        )

        # Normalization (MinMaxScaler)
        # Normalizing feature values to the range [0, 1] using Min-Max scaling.
      #the scaler on training data and apply the same transformation to test data.
        scaler = MinMaxScaler()
        self.train_x = scaler.fit_transform(self.train_x)
        self.test_x = scaler.transform(self.test_x)
        # For Multi-Layer Perceptron (MLP), one-hot encode the labels if there are 3 or more classes.


        # One-hot encode if multi-class (for MLP)
        self.train_y_onehot = self.train_y
        self.test_y_onehot = self.test_y
        if len(np.unique(self.data_y)) >= 3:
            encoder = LabelBinarizer()
            self.train_y_onehot = encoder.fit_transform(self.train_y)
            self.test_y_onehot = encoder.transform(self.test_y)
        # <==== insert your code above here

    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.

        For each of the algorithms KNearest Neighbour, DecisionTreeClassifer and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination, 
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and  lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.

        """
        # ====> insert your code below here
        for k in [1, 3, 5, 7, 9]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train_x, self.train_y)
            accuracy = model.score(self.test_x, self.test_y)
            self.stored_models["KNN"].append(model)
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = len(self.stored_models["KNN"]) - 1

        # Decision Tree hyperparameters
        for depth in [1, 3, 5]:
            for min_split in [2, 5, 10]:
                for min_leaf in [1, 5, 10]:
                    model = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y)
                    accuracy = model.score(self.test_x, self.test_y)
                    self.stored_models["DecisionTree"].append(model)
                    if accuracy> self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = len(self.stored_models["DecisionTree"]) - 1

        # MLP hyperparameters 3TRAINING MODEL
        for n1 in [2, 5, 10]:
            for n2 in [0, 2, 5]:
                for activation in ["logistic", "relu"]:
                    if n2 == 0:
                        layers = (n1,)
                    else:
                        layers = (n1, n2)
                    model = MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation=activation,
                        max_iter=1000,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y_onehot)
                    #CHECKING THE accuracy
                    accuracy= model.score(self.test_x, self.test_y_onehot)
                    #storing the accuracy
                    self.stored_models["MLP"].append(model)
                    #updating the accuracy
                    if accuracy> self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = len(self.stored_models["MLP"]) - 1

        # <==== insert your code above here

    def report_best(self) :
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"

        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        best_algorithm = max(self.best_accuracy, key=self.best_accuracy.get)   #finding teh highest accuracy
        best_index = self.best_model_index[best_algorithm] #getting index for best mdoel
        best_model = self.stored_models[best_algorithm][best_index] #finding best model from stored model
        best_acc = self.best_accuracy[best_algorithm] #retriving best accurcy value
        return best_acc, best_algorithm, best_model 

        # <==== insert your code above here

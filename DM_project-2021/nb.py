import numpy as np

# MAHYA
def train_nb(X, y):
    """
    Train the Naive Bayes classifier. For NB this is just
    computing the necessary probabilities to perform classification
    1. The probability P(ci) for every class -> prior (the prior comes from the distribution of labels)
    2. The mean and std -> mean, std (The mean and variance are applied to each feature in the input data X)

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
    consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
        y[i] is the label for X[i].

    Outputs:
    - prior : list with length equal to the number of classes
    - mean : A numpy array of shape (num_classes, num_features)
    - std  : A numpy array of shape (num_classes, num_features)

    **** train() should be run with X as training data
    """
    # use list comprehension

    # Separate training points by class
    ClassLabels, count_Class = np.unique (y, return_counts=True)
    DataInClass = [[x for x, t in zip (X, y) if t == c] for c in ClassLabels]  # categorize for each class

    #########################################################################
    # TODO:                                                                 #
    # compute class prior                                                   #
    #########################################################################

    Number_of_InputData = y.size
    prior = count_Class / Number_of_InputData
    
    #########################################################################
    # Detect type of our data                                               #
    #   Categorical or continuous                                           #
    #########################################################################

    Col_Continue = np.array([np.all([isinstance(_xj, (int, float)) for _xj in cols])
                             for cols in X.T])
    IScontinuous = np.all(Col_Continue) 
    
    #########################################################################
    # TODO:                                                                 #
    # Estimate mean and std for each class / feature                        #
    #########################################################################
    if IScontinuous:
        # mean and std for continuous data
        
        mean = np.array([np.mean(x, axis=0) for x in DataInClass])
        std = np.array([np.std(x, axis=0) for x in DataInClass])
        
        # mean is a c x d array : c=number of classes and d is dimension of X
        # std is a c x d array : c=number of classes and d is dimension of X

        result = prior, mean, std
    else:
        # frequancy for categorical data
        frequancy = []
        for i, x in enumerate(DataInClass):
            x1 = np.array(x)
            frequancy.append([{feature: count / count_Class[i] for feature, count in zip(*np.unique(cols, return_counts=True))}
                 for cols in x1.T])
        result = prior, frequancy, 
    return result


# Normal Distribution
def normal_distribution(x, mean, std):
    """
    Compute normal distribution
    output size: (num_input_data, num_features)

    """
    #########################################################################
    # TODO : Compute normal distribution                                    #
    #########################################################################

    normal = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))

    return normal


def predict(X, prior, mean=None, std=None, freq=None, targets=None):
    """
    Using the dustributions from before, predict labels for test data (or train data) using this classifier.
    We predict the class of the data maximizing the likelihood or you can
     maximize the log likelihood to make it numericaly more stable.
     (This is possible since f(x)=log(x) is a monotone function)

    You have to compute:
    - Compute the conditional probabilities  P(x|c) (or log P(x|c) )
    - The posterior (if you compute the log likelihood the product  becomes sum)
    - Make the prediction

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
        of num_test samples each of dimension D.
    - prior, mean, std: output of train() function

    Returns:
    - y_pred : A numpy array of shape (num_test,) containing predicted labels for the
    test data, where y[i] is the predicted label for the test point X[i].

    *** predict() should be run with X as test data, based on mean and variance and prior from the training data
        (to compute the training accuracy run with X as train data)

    """
    
    Col_Continue = np.array([np.all([isinstance(_xj, (int, float)) for _xj in cols])
                             for cols in X.T])
    IScontinuous = np.all(Col_Continue)
    # use list comprehension

    #################################################################################
    #        # Compute the conditional probabilities  P(x|c)                        #
    #             # There are three loops in the code.                              #
    #             # 1. through each sample.                                         #
    #             # 2. through each class.                                          #
    #             # 3. through each attribute and apply the Normal/ logNormal distribution. #
    #        # Compute the posterior                                                #
    #                                                                               #
    #################################################################################


    #########################################################################
    #                           TODO
    #             compute the posterior and predict                         #
    # - hint for prediction: class having the biggest probability[argmax()] #
    #########################################################################

    if IScontinuous:
        
        Log_Likelihood = np.array([np.sum(np.log(normal_distribution(x, mean, std)), axis=1) for x in X])
        Log_Posterior = Log_Likelihood + np.log(prior)
        y_pred = np.argmax(Log_Posterior, axis=1)
        return y_pred
    else:
        # frequancy for categorical data
        log_likelihood = np.array(
            [np.array([np.sum([np.log(f[feature]) for feature, f in zip(x, features_f)]) for x in X])
             for features_f in freq]).T

        log_posterior = log_likelihood + np.log(prior)

        y_pred = np.argmax(log_posterior, axis=1)
        
        if targets is not None:
            return targets[y_pred]
        else:
            return y_pred

    # posterior =

    # y_pred =
   


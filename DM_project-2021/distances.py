import numpy as np

#compute_euclidean_dist_two_loops................................................................
def compute_euclidean_dist_two_loops(x_train, x_test):
    """
    Compute the distance between each test point in x_test and each training point
    in x_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - x_train: A numpy array of shape (num_train, D) containing test data.
    - x_test: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
        #####################################################################
        #Compute the l2 distance between the ith test point and the jth     #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
            Diff = x_test[i,:] - x_train[j,:]
            dists[i,j] = np.sqrt(np.sum(Diff**2))
    return dists

#compute_euclidean_dist_one_loop...................................................................
def compute_euclidean_dist_one_loop(x_train, x_test):
    """
    Compute the distance between each test point in x_test and each training point
    in x_train using a single loop over the test data.

    Input / Output: Same as compute_euclidean_dist_two_loops
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # dists[i,:] = np.diag(np.dot((x_test[i,:] - x_train),(x_test[i,:] - x_train).T))
      dists[i,:] = np.sqrt(np.sum((x_test[i,:] - x_train)**2, axis=1))
    return dists


#compute_euclidean_dist_no_loops......................................................................
def compute_euclidean_dist_no_loops(x_train, x_test):
    """
    Compute the distance between each test point in x_test and each training point
    in x_train using no explicit loops.

    Input / Output: Same as compute_euclidean_dist_two_loops
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    #Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #  X = np.diag(np.dot(x_test,x_test.T))   dot operator take time
    X = np.sum(x_test**2,axis=1)
    X_TEST = X.reshape((X.shape[0],1))
    # X_TEST = np.repeat(X_TEST,num_train,axis=1)    not necessary

    #  Y = np.diag(np.dot(x_train,x_train.T))         np.dot increase processing time
    Y = np.sum (x_train**2,axis=1)
    X_TRAIN = np.reshape(Y,(1,Y.shape[0]))
    # X_TRAIN = np.repeat(X_TRAIN.T,num_test,axis=0)    

    dists = np.sqrt(X_TEST- 2* np.dot(x_test, x_train.T) + X_TRAIN)
    return dists


#compute_mahalanobis_dist........................................................................
def compute_mahalanobis_dist(x_train, x_test, sigma):
    """
    Compute the Mahalanobis distance between each test point in x_test and each training point
    in x_train (please feel free to choose the implementation).

    Inputs:
    - x_train: A numpy array of shape (num_train, D) containing test data.
    - x_test: A numpy array of shape (num_test, D) containing test data.
    - sigma: A numpy array of shape (D,D) containing a covariance matrix.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Mahalanobis distance between the ith test point and the jth training
      point.
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # Compute the Mahalanobis distance between all test points and all      #
    # training points (please feel free to choose the implementation),      #
    # and store the result in dists.                                        #
    #########################################################################
    Inv_Sigma = np.linalg.inv(sigma)
    for i in range(num_test):
        for j in range(num_train):
            Diff = np.dot(Inv_Sigma,(x_test[i,:] - x_train[j,:]))
            dists[i,j] = np.sqrt(np.dot(Diff,(x_test[i,:] - x_train[j,:])))
    return dists

#compute_manhattan_dist............................................................................
def compute_manhattan_dist(x_train, x_test):
    """
    Compute the Manhattan distance between each test point in x_test and each training point
    in x_train (please feel free to choose the implementation).

    Inputs:
    - x_train: A numpy array of shape (num_train, D) containing train data.
    - x_test: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Manhattan distance between the ith test point and the jth training
      point.
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    #  Compute the Manhattan distance between all test points and all      #
    # training points (please feel free to choose the implementation),
    # and store the result in dists. #                                     #
    #########################################################################
    for i in range(num_test):
        dists[i,:] = np.sum(np.abs(x_test[i,:] - x_train), axis=1)             
    return dists


#define_covariance...............................................................................
def define_covariance(X_train, method):
    """
    Define a covariance  matrix using 3 difference approaches: 
    """
    d = X_train.shape[1]

    #########################################################################
    # Computre Σ as a diagonal matrix that has at its diagonal the average  #
    # variance of  the different features,       
    #  i.e. all diagonal entries Σ_ii will be the same                      #
    #                                                                       #
    #########################################################################
    if method == 'diag_average_cov':
        cov = np.eye(d) * np.diag(np.cov(X_train)).mean()
        return cov
  
    #########################################################################
    # Computre  Σ as a diagonal matrix that has at its diagonal             #
    # he variance of eachfeature, i.e.σ_k     
    #                                                                       #
    #########################################################################
   
    elif method == 'diag_cov':
        cov = np.diag(np.diag(np.cov(X_train.T))) 
        return cov

    #########################################################################
    # Computre Σ as the full covariance matrix between all pairs of features#
    #                                                                       #
    #########################################################################

    # Your code
    elif method == 'full_cov':
        cov = np.cov(X_train.T)
        return cov
    else:
        raise ValueError("Uknown method identifier.")







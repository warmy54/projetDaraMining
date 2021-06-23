from __future__ import print_function, division
import numpy as np
import math

class LinearRegression_RidgeRegression():
    """Linear Regression and Ridge Regression.
    Parameters:
    -----------
    X: data
    y: target values y
    iterations: int
         number of training iterations
    lr: float
        the learning rate
    l2_reg: float
        parameter for l2 regularizer
        l2_reg = 0 ->  Linear Regression
        l2_reg != 0 ->  Ridge Regression
    analytical_sol: boolean
        True or false depending if analytical solution will be used during the training.
    """
    def __init__(self, X, y, iterations=100, lr=0.001, l2_reg = 0, analytical_sol=True, SGD = False, BatchNumber=5):
        self.SGD = SGD
        self.BatchNumber = BatchNumber
        self.analytical_sol = analytical_sol
        self.iterations = iterations
        self.lr = lr
        self.l2_reg = l2_reg
        self.X = X
        self.y = y
        self.n_features = self.X.shape[1]
        """ Initialize weights randomly [-1/d, 1/d] """
        limit = 1 / np.sqrt(self.n_features)
        self.w = np.random.uniform(-limit, limit, (self.n_features, 1))

    def fit(self):
        """Function that returns the weights of Linear Regression and
        Ridge Regression(Linear Regression with l2 regularizer)
        for both analytical solution and applying optimization method.
        If analytical_sol == 0 returns the weights computed using the analytical solution
        otherwise returns the weights computed using optimization.
        """
        # If analytical_sol => Least squares
        if self.analytical_sol:
            ######################################################################
            # To do:                                                             #
            # for both Linear and Ridge Regression (Linear with l2 regularizer)  #
            # Calculate weights by least squares (analytical solution)           #
            ######################################################################
            _X = self.X
            XPrime = np.concatenate((_X,np.ones((_X.shape[0],1))),axis=1)           
            XT = np.transpose(XPrime)
            XT_X = np.dot(XT, XPrime)
            # print("Since the matrix determinat is", np.linalg.det(XT_X), ", So it isnot singular")
            Inv_XT_X = np.linalg.pinv(XT_X+ self.l2_reg * np.eye(XT_X.shape[0]))
            self.w = np.dot(np.dot(Inv_XT_X,XT),self.y)
            return self.w
        else:
            ######################################################################
            # To do:                                                             #
            # for both Linear and Ridge Regression (Linear with l2 regularizer)  #
            # Calculate weights using gradient descent (GD)                      #
            ######################################################################
            _X = self.X
            w_old = np.concatenate((self.w,np.zeros((1,1))))
            XPrime = np.concatenate((_X,np.ones((_X.shape[0],1))),axis=1) 
            #print(XPrime.shape)
            Y = self.y
            y_pred = np.dot(XPrime, w_old)
            print(y_pred.shape)
            if self.SGD:
                Bt = int(np.floor(Y.shape[0]/self.BatchNumber))
                for k in range(self.iterations):
                    for i in range(self.BatchNumber):
                        S = Bt*i
                        E = Bt*(i+1)-1
                        y_pred[S:E,] = np.dot(XPrime[S:E,:], w_old)
                        Delta =   -2*self.lr *(-np.dot(XPrime[S:E,].T,( Y[S:E,]- y_pred[S:E,]))+self.l2_reg*w_old)
                        w_new = w_old + Delta
                        w_old = w_new[:]
                    
                self.w = w_new
            else:
                for k in range(self.iterations):
                    Delta =   -2*self.lr * (-np.dot(XPrime.T,( Y- y_pred))+self.l2_reg*w_old)
                    w_new = w_old + Delta
                    w_old = w_new[:]
                    y_pred = y_pred = np.dot(XPrime, w_old)
                self.w = w_new
            return self.w

    def predict(self, X):

        ######################################################################
        # To do:                                                             #
        # make prediction                                                    #
        ######################################################################
        y_pred = np.zeros(X.shape[0],)
        XPrime = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
        y_pred = np.dot(XPrime,self.w)
        return y_pred



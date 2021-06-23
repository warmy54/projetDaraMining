from regression import LinearRegression_RidgeRegression
import numpy as np


# usage in main part: cross_validation( Normal_X_train_valid_IC, y_train, Kfold = 10)

def cross_validation(X_train, y_train, Analytic_Sol=True, sgd=False, Kfold = 10, l2r =10000):

    y_tr_val = y_train[:,0]
    y_tr_val = np.reshape(y_tr_val,(y_tr_val.shape[0],1)) 


    # Normal_X_train_valid_IC = Normal_X_train_IC

    n_train_valid = y_tr_val.shape[0]

    n_Xval_parts = Kfold   # 10 fold cross validation 

    Xval_part_size = (int)(n_train_valid/n_Xval_parts)

    best_validation_MSE = 1000000

    i = 1
    for val_start in range(0,n_train_valid,Xval_part_size):
    #     print('part: ', i)
        val_end = val_start + Xval_part_size
    #     print('start: ', val_start, ' , end: ', val_end)
        y_val = y_tr_val[val_start:val_end,0]
        Normal_X_valid_IC = X_train[val_start:val_end,:]

        y_tr = np.concatenate((y_tr_val[0:val_start,0],y_tr_val[val_end:n_train_valid,0]))
        Normal_X_train_IC = np.concatenate((X_train[0:val_start,:],X_train[val_end:n_train_valid,:]))
    #     print(y_tr.shape)

        y_tr = np.reshape(y_tr,(y_tr.shape[0],1)) # we change the shape of y_tr from (m,) to (m,1)


        model=LinearRegression_RidgeRegression(Normal_X_train_IC, y_tr, iterations=500,lr=0.0001,l2_reg=l2r, 
                                                  analytical_sol=Analytic_Sol, SGD=sgd, BatchNumber= int(Normal_X_train_IC.shape[0]/10))


    #     print('Normal_X_train_IC shape: ',Normal_X_train_IC.shape)
    #     print('y_train shape: ',y_train.shape)
        w = model.fit()



        y_pred = model.predict(Normal_X_valid_IC)
        y_pred = y_pred.reshape(y_pred.shape[0]) # reduce dimention  
    #     print('y_pred : ', y_pred.shape)


        y_val = np.reshape(y_val,(y_val.shape[0],1))
        y_val= y_val.flatten()  # reduce dimension for MSE cacule 
    #     print('y_val: ',y_val.shape)



        MSE = abs (np.sum((y_pred-y_val)**2)) / y_val.shape[0]
        print("\nThe Validation MSE is : %0.5E" %(MSE) )

        if (MSE < best_validation_MSE):
            # print('found better!')
            best_validation_MSE = MSE
            best_validation_model = model

        i += 1
    #     print('y_val: ', y_val)
    #     print('y_tr: ', y_tr)
        if (i > n_Xval_parts):
            break
        return model


   


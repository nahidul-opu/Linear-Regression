import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self, max_iteration = 10000, max_mse = None, patience = 5,learning_rate = 0.001,threshold = 10):
        self.max_iteration=max_iteration
        self.max_mse = max_mse
        self.patience = patience
        self.learning_rate = learning_rate
        self.threshold = threshold
        return 

    def fit(self, X, Y):
        if len(X)!=len(Y):
            raise ValueError("Data and Label Size Must Be Same")
        if isinstance(X,pd.Series):
            X = X.to_frame()
        Y = Y.to_frame()
        self.n = len(X)
        self.coeff = [0 for _ in range(len(X.columns))]
        self.intercept = 0
        self.mse=[]
        self.n_iteration = 0
        X = X.values
        Y = Y.values
        while(True):
            y_pred = np.sum(X*self.coeff,axis=1) + self.intercept
            y_pred = y_pred.reshape(self.n,1)
            current_mse = np.square(np.subtract(Y,y_pred)).mean()
            self.mse.append(current_mse)
            #Dm = (-2/self.n)*np.sum(X*(Y-y_pred).reshape(self.n,1),axis=0)
            #Dc = ((-2/self.n)*np.sum(Y-y_pred))
            Dm = -2*(X*(Y-y_pred).reshape(self.n,1)).mean(axis=0)
            Dc = -2*(Y-y_pred).mean()
            self.coeff = self.coeff - Dm*self.learning_rate
            self.intercept = self.intercept - Dc*self.learning_rate
            self.n_iteration = self.n_iteration + 1
            if self.max_mse==None:
                if self.n_iteration >= self.max_iteration:
                    break
            else:
                if self.check_for_break():
                    break
        
    def check_for_break(self):
        if abs(self.mse[-1])<=self.max_mse:
            return True
        elif len(self.mse)<self.patience:
             return False
        else:
            mse0 = self.mse[-self.patience]
            mse1 = self.mse[-1]
            if abs(abs(mse0)-abs(mse1))<=self.threshold:
                return True
            else:
                return False

    def predict(self,X):
        if isinstance(X,list):
            X = [X]
        if isinstance(X,pd.Series):
            X = X.to_frame()
        if isinstance(X,pd.DataFrame):
            X = X.values
        n = len(X)
        return (np.sum(X*self.coeff,axis=1) + self.intercept).reshape(n,1)
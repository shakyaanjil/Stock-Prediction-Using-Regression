import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style
import math

class OrdinaryLeastSquares(object):
    
    def __init__(self):
            self.coefficients =[]
        
    def fit(self, X, y):
        if len(X.shape) == 1:X = self._reshape_x(X)
        
        X=self._concatenate_ones(X)
        self.coefficients=np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    
    def predict(self,entry):
        b0 = self.coefficients[0]
        other_betas = self.coefficients[1:]
        prediction = b0
        
        for xi,bi in zip(entry,other_betas):
            prediction += (bi * xi)

        return prediction
    
        
    def _reshape_x(self,X):
        return X.reshape(-1,1)
    
    def _concatenate_ones(self,X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        return np.concatenate((ones,X),1)  


def main():
    df = pd.read_csv('google.csv')
    st.title("Stock Prediction Web App")
    st.sidebar.title("Stock Prediction Web App")
    st.markdown("There is a risk in everything, so be prepared for the ups and downs.")

    x = df[['High','Open','Low','Volume']].values
    # print(x.shape)
    y = df['Close'].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    regressor = OrdinaryLeastSquares()
    regressor.fit(x_train,y_train)

    y_pred = []
    for row in x_test:
      y_pred.append(regressor.predict(row))
    y_pred=np.array(y_pred)
    result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
    print(result)

    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    High=st.sidebar.number_input("High price")
    Open=st.sidebar.number_input("Open price")
    Low=st.sidebar.number_input("Low price")
    Volume=st.sidebar.number_input("Volume price")
    
    user_input=[High,Open,Low,Volume] 
    
    if st.sidebar.button("Predict", key='Predict'):
        st.subheader("Stock Prediction Results")
        regressor.fit(x_train,y_train)
        st.write("Prediction: ", regressor.predict(user_input))
        graph = result.head(20)
        graph.plot()
        st.pyplot()


if __name__ == '__main__':
    main()

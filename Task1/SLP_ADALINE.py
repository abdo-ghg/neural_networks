import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

df = pd.read_csv('penguins.csv')
def preprocessing(features_selected, classes_selected):
    new_df = df[df['Species'].isin(classes_selected)][['Species'] + [df.columns[i] for i in features_selected]]

    X = new_df.drop(columns = 'Species')
    y = new_df['Species'].values

    num_cols = X.select_dtypes(include = np.number).columns
    cat_cols = X.select_dtypes(exclude = np.number).columns

    imputer = KNNImputer()
    X[num_cols] = imputer.fit_transform(X[num_cols])

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    transformer = ColumnTransformer([('num', 'passthrough', num_cols), ('cat', OneHotEncoder(), cat_cols)])
    X = transformer.fit_transform(X)

    y = np.where(y == classes_selected[0], 1, -1)
    return X, y
  
def custom_train_test_split(X, y):
    class_1 = np.random.permutation(np.where(y == 1)[0])
    class_2 = np.random.permutation(np.where(y == -1)[0])
    train_idx = np.random.permutation(np.r_[class_1[:30], class_2[:30]])
    test_idx  = np.random.permutation(np.r_[class_1[30:], class_2[30:]])
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test

def SLP_train(X_train, y_train, epochs, learning_rate, X0):
  w0 = np.random.rand()
  w1 = np.random.rand()
  w2 = np.random.rand()
  for epoch in range(epochs):
    for i in range(len(X_train)):
      net = (X_train[i][0] * w1) + (X_train[i][1] * w2) + (X0 * w0)
      if(net >= 0):
        y_pred = 1
      else:
        y_pred = -1
      e = y_train[i] - y_pred
      w0 = w0 + learning_rate * e * X0
      w1 = w1 + learning_rate * e * X_train[i][0]
      w2 = w2 + learning_rate * e * X_train[i][1]
  return w0, w1, w2

def SLP_test(X_test, y_test, X0, w0, w1, w2):
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  for i in range(len(X_test)):
      net = (X_test[i][0] * w1) + (X_test[i][1] * w2) + (X0 * w0)
      if(net >= 0):
        y_pred = 1
      else:
        y_pred = -1
      if((y_pred == 1) & (y_test[i] == 1)):
        TP +=1
      elif((y_pred == -1) & (y_test[i] == -1)):
        TN +=1
      elif((y_pred == 1) & (y_test[i] == -1)):
        FP +=1
      else:
        FN +=1
  P = TP + FN
  N = TN + FP
  accuracy = (TP + TN) / (P + N) if (P + N) > 0 else 0
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  conf_mat = [[TP, FP],
              [FN, TN]]
  return accuracy, precision, recall, F1, conf_mat



def ADA_train(X_train, y_train, epochs, learning_rate, X0, mse):
  w0 = np.random.rand()
  w1 = np.random.rand()
  w2 = np.random.rand()
  for epoch in range(epochs):
    for i in range(len(X_train)):
      net = (X_train[i][0] * w1) + (X_train[i][1] * w2) + (X0 * w0)
      y_pred = net
      e = y_train[i] - y_pred
      w0 = w0 + learning_rate * e * X0
      w1 = w1 + learning_rate * e * X_train[i][0]
      w2 = w2 + learning_rate * e * X_train[i][1]
    e = 0
    for i in range(len(X_train)):
        net = (X_train[i][0] * w1) + (X_train[i][1] * w2) + (X0 * w0)
        y_pred = net
        e += y_train[i] - y_pred
    MSE = (1 / (2 * len(X_train))) * (e ** 2)
    if MSE < mse:
        break
  return w0, w1, w2

def ADA_test(X_test, y_test, X0, w0, w1, w2):
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  for i in range(len(X_test)):
      net = (X_test[i][0] * w1) + (X_test[i][1] * w2) + (X0 * w0)
      if(net >= 0):
        y_pred = 1
      else:
        y_pred = -1
      if((y_pred == 1) & (y_test[i] == 1)):
        TP +=1
      elif((y_pred == -1) & (y_test[i] == -1)):
        TN +=1
      elif((y_pred == 1) & (y_test[i] == -1)):
        FP +=1
      else:
        FN +=1
  P = TP + FN
  N = TN + FP
  accuracy = (TP + TN) / (P + N) if (P + N) > 0 else 0
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  conf_mat = [[TP, FP],
              [FN, TN]]
  return accuracy, precision, recall, F1, conf_mat

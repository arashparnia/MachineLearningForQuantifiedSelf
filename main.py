# from pandas import scatter_matrix
# from pandas.io.parsers import read_csv
# from rpy2.robjects import r, pandas2ri
# from sklearn import preprocessing, model_selection
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
#
# from boruta import BorutaPy
# from pprint import pprint
# from matplotlib import interactive
#
#
# import matplotlib.pyplot as plt
# import rpy2.robjects as robjects
# # import pandas.rpy.common as com
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn.cluster as cluster
# import time
#
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier





print("main")

# data = pd.read_csv('../Data/csv-participant-one/accelerometer_phone.csv')
# # data = pd.read_csv('../Data/csv-participant-one/accelerometer_phone.csv',nrows = 1000)
# print(data.head())
# plt.hist(data['x'])
# plt.hist(data['y'])
# plt.hist(data['z'])
# # plt.scatter(data['y'], data['z'])
# plt.show()

import PythonCode.Chapter2.CreateDataset
db= PythonCode.Chapter2.CreateDataset.CreateDataset('../Data/csv-participant-one',0)
print(db)
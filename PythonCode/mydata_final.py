import datetime
import time
import pandas as pd
from matplotlib import pylab
from pandas import scatter_matrix
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
from pprint import pprint

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn.cluster as cluster
# import time
#
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

import copy
import os

from scipy.stats import normaltest, shapiro, stats

from PythonCode.Chapter2.CreateDataset import CreateDataset
from PythonCode.util import util
from PythonCode.util.VisualizeDataset import VisualizeDataset


from Chapter2.CreateDataset import CreateDataset
from util import util
from util.VisualizeDataset import VisualizeDataset

dataset_path = '../Data/walking/'
result_dataset_path = './intermediate_datafiles/'
#
# myDate = '2017-06-23 15:42:42.144 +0200'
# myDate =  myDate.rsplit('+', 5)[0]
# print (myDate)
# dt = datetime.datetime.strptime(myDate, "%Y-%m-%d %H:%M:%S.%f ")
# unixtime =  time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
# unixtime = unixtime * 1000000
#
# print (unixtime)
# print(dt.microsecond )
# print(dt.microsecond / 1000000.0)
# exit(0)


print("main")

def makeDataFile():
    # loading original csv file
    # datafile = pd.read_csv(dataset_path + 'arm.csv',nrows = 10)
    dataset_ankle = pd.read_csv(dataset_path + 'ankle.csv')
    # dataset_ankle['loggingTime(txt)'] = pd.DatetimeIndex(dataset_ankle['loggingTime(txt)'])
    # dataset_ankle.index = dataset_ankle['loggingTime(txt)']

    dataset_waist = pd.read_csv(dataset_path + 'waist.csv')
    # dataset_waist['loggingTime(txt)'] = pd.DatetimeIndex(dataset_waist['loggingTime(txt)'])
    # dataset_waist.index = dataset_waist['loggingTime(txt)']


    # print(min(dataset_ankle['accelerometerTimestamp_sinceReboot(s)']))
    # print(max(dataset_ankle['accelerometerTimestamp_sinceReboot(s)']))
    unixDateTime_waist = []
    for index, row in dataset_waist.iterrows():

        myDate = row['loggingTime(txt)'] #'2017-06-23 15:42:42.144 +0200'
        myDate =  myDate.rsplit('+', 5)[0]
        # print (myDate)
        dt = datetime.datetime.strptime(myDate, "%Y-%m-%d %H:%M:%S.%f ")
        unixtime =  time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
        # unixtime = unixtime * 1000000
        # print(unixtime)

        unixDateTime_waist.append(unixtime)
    # print (unixDateTime)


    dataset_waist['unixDateTime_waist'] = unixDateTime_waist
    dataset_waist.rename(columns={'loggingTime(txt)': 'time_waist'},inplace=True)
    dataset_waist.rename(columns={'state(N)': 'state'},inplace=True)

    dataset_waist.rename(columns={'accelerometerAccelerationX(G)': 'acc_waist_x'},inplace=True)
    dataset_waist.rename(columns={'accelerometerAccelerationY(G)': 'acc_waist_y'},inplace=True)
    dataset_waist.rename(columns={'accelerometerAccelerationZ(G)': 'acc_waist_z'},inplace=True)


    dataset_waist.rename(columns={'gyroRotationX(rad/s)': 'gyr_waist_x'},inplace=True)
    dataset_waist.rename(columns={'gyroRotationY(rad/s)': 'gyr_waist_y'},inplace=True)
    dataset_waist.rename(columns={'gyroRotationZ(rad/s)': 'gyr_waist_z'},inplace=True)

    unixDateTime_ankle = []
    for index, row in dataset_ankle.iterrows():
        myDate = row['loggingTime(txt)']  # '2017-06-23 15:42:42.144 +0200'
        myDate = myDate.rsplit('+', 5)[0]
        # print (myDate)
        dt = datetime.datetime.strptime(myDate, "%Y-%m-%d %H:%M:%S.%f ")
        unixtime = time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
        # unixtime = unixtime * 1000000
        # print(unixtime)

        unixDateTime_ankle.append(unixtime)

    dataset_ankle['unixDateTime_ankle'] = unixDateTime_ankle
    dataset_ankle.rename(columns={'loggingTime(txt)': 'time_ankle'}, inplace=True)
    dataset_ankle.rename(columns={'accelerometerAccelerationX(G)': 'acc_ankle_x'},inplace=True)
    dataset_ankle.rename(columns={'accelerometerAccelerationY(G)': 'acc_ankle_y'},inplace=True)
    dataset_ankle.rename(columns={'accelerometerAccelerationZ(G)': 'acc_ankle_z'},inplace=True)


    dataset_ankle.rename(columns={'gyroRotationX(rad/s)': 'gyr_ankle_x'},inplace=True)
    dataset_ankle.rename(columns={'gyroRotationY(rad/s)': 'gyr_ankle_y'},inplace=True)
    dataset_ankle.rename(columns={'gyroRotationZ(rad/s)': 'gyr_ankle_z'},inplace=True)

    # print (list(dataset_ankle))
    # print (list(dataset_waist))
    # exit(0)
    df_waist = dataset_waist[['unixDateTime_waist','time_waist','state','acc_waist_x','acc_waist_y','acc_waist_z','gyr_waist_x','gyr_waist_y','gyr_waist_z']].copy()
    df_ankle = dataset_ankle[['unixDateTime_ankle','time_ankle','acc_ankle_x','acc_ankle_y','acc_ankle_z','gyr_ankle_x','gyr_ankle_y','gyr_ankle_z']].copy()
    #
    #
    #
    # DataViz = VisualizeDataset()
    # DataViz.plot_dataset(df_waist, ['acc_', 'gyr_'],
    #                      ['like', 'like'],
    #                      ['line', 'line'])
    #
    #
    # DataViz.plot_dataset(df_ankle, ['acc_', 'gyr_'],
    #                      ['like', 'like'],
    #                      ['line', 'line'])
    dataset = pd.concat([df_waist, df_ankle],axis=1)

    # dataset = pd.merge(df_waist,df_ankle, how='outer',left_on='unixDateTime_waist', right_on='unixDateTime_ankle')
    # print (dataset)
    # df_waist.to_csv(dataset_path + 'data_waist.csv')
    # df_ankle.to_csv(dataset_path + 'data_ankle.csv')
    # dataset = df_waist.join(df_ankle).copy()
    # # dataset = pd.join(df_waist,df_ankle)
    #
    print (list(dataset))
    print (dataset)
    #
    dataset.to_csv(dataset_path + 'data.csv')
    #


def makeLabelFile():
    dataset = pd.read_csv(dataset_path + 'labels.csv',index_col=0)

    unixDateTime = []
    for index, row in dataset.iterrows():

        myDate = row['label_start_datetime'] #'2017-06-23 15:42:42.144 +0200'
        myDate =  myDate.rsplit('+', 5)[0]
        # print (myDate)
        dt = datetime.datetime.strptime(myDate, "%Y-%m-%d %H:%M:%S.%f ")
        unixtime =  time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
        # unixtime = unixtime * 1000000
        # print(unixtime)

        unixDateTime.append(unixtime)

    dataset['start_datetime'] = unixDateTime

    unixDateTime = []
    for index, row in dataset.iterrows():

        myDate = row['label_end_datetime'] #'2017-06-23 15:42:42.144 +0200'
        myDate =  myDate.rsplit('+', 5)[0]
        # print (myDate)
        dt = datetime.datetime.strptime(myDate, "%Y-%m-%d %H:%M:%S.%f ")
        unixtime =  time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
        # unixtime = unixtime * 1000000
        # print(unixtime)

        unixDateTime.append(unixtime)

    dataset['end_datetime'] = unixDateTime

    dataset.to_csv(dataset_path + 'labels.csv')


    # dataset_waist = pd.read_csv(dataset_path + 'waist.csv')
    # # MAKING LABELS ---------------------------------------------------------
    # labeldf = pd.DataFrame()
    # start_time = dataset_waist['loggingTime(txt)'][1]
    # start_time_since1970 = dataset_waist['accelerometerTimestamp_sinceReboot(s)'][1]
    #
    # currentstate = 0
    # for index, row in dataset_waist.iterrows():
    #     if (currentstate != row['state(N)']):
    #
    #         end_time = (row['loggingTime(txt)'])
    #         print('--------------------------------------------------')
    #         print(start_time)
    #         print (currentstate)
    #         print(end_time)
    #         print('--------------------------------------------------')
    #         if (currentstate == 0): thelabel = 'standing'
    #         elif (currentstate == 1): thelabel = 'indoorWalking'
    #         elif (currentstate == 2): thelabel = 'OutdoorWalkingStone'
    #         elif (currentstate == 3): thelabel = 'OutdoorWalkingGrass'
    #         elif (currentstate == 4): thelabel = 'OutdoorWalkingSand'
    #         elif (currentstate == 5): thelabel = 'onTheTable'
    #
    #         labeldf = labeldf.append({'sensor_type': 'interval_label',
    #                                    'device_type': 'smartphone' ,
    #                                    'label': thelabel,
    #                                    'label_start_datetime': start_time,
    #                                    'label_end_datetime' : end_time,
    #                                     'state' : currentstate
    #                                    }, ignore_index=True)
    #
    #         start_time = (row['loggingTime(txt)'])
    #         currentstate = row['state(N)']
    #
    # # print (labeldf)
    # labeldf.to_csv(dataset_path + 'labels.csv')






def makeDataSet(granularities):
    datafile = 'data.csv'
    # datafile_waist = 'data_waist.csv'
    labelfile = 'labels.csv'

    # Chapter 2: Initial exploration of the dataset.

    # Set a granularity (i.e. how big are our discrete time steps). We start very
    # coarse grained, namely one measurement per minute, and secondly use four measurements
    # per second

    datasets = []
    milliseconds_per_instance =granularities
    print ('granularities ' + str(milliseconds_per_instance))
    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(dataset_path, milliseconds_per_instance)

    # Add the selected measurements to it.
    DataSet.add_numerical_dataset(datafile, 'time_waist',
                                  ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'],
                                  'avg', '')

    DataSet.add_numerical_dataset(datafile, 'time_waist',
                                  ['gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z'],
                                  'avg','')

    DataSet.add_numerical_dataset(datafile, 'time_waist',
                                  ['acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z'],
                                  'avg', '')

    DataSet.add_numerical_dataset(datafile, 'time_waist',
                                  ['gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'],
                                  'avg', '')

    DataSet.add_event_dataset(labelfile, 'label_start_datetime', 'label_end_datetime', 'label', 'binary')


    dataset = DataSet.data_table

    # Plot the data

    DataViz = VisualizeDataset()

    # Boxplot
    # DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z'])

    # Plot all data

    DataViz.plot_dataset(dataset, ['acc_', 'gyr_','label'],
                         ['like', 'like','like'],
                         ['line', 'line','points'])
    # And print a summary of the dataset

    # util.print_statistics(dataset)
        # datasets.append(copy.deepcopy(dataset))

    # Finally, store the last dataset we have generated (250 ms).
    dataset.to_csv(result_dataset_path + 'mydata_final_chapter2_result.csv')
    print ('done')




def normalityCheck():

    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter2_result.csv',index_col=0)
    dataset.index = dataset.index.to_datetime()
    normaltest(dataset['acc_phone_x'])
    shapiro(dataset['acc_phone_x'])

    plt.subplot(3,3,1)
    plt.title('acc_phone_x')
    stats.probplot(dataset['acc_phone_x'], dist="norm", plot=pylab)

    pylab.show()









makeDataFile()
# makeLabelFile()
makeDataSet(250)
# normalityCheck()


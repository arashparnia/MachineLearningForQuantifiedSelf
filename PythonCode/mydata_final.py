import copy
import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pylab

from Chapter2.CreateDataset import CreateDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter5.Clustering import HierarchicalClustering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from util import util
from util.VisualizeDataset import VisualizeDataset

from pandas.io.parsers import read_csv
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import time

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


    dataset_waist = pd.read_csv(dataset_path + 'waist.csv')
    # MAKING LABELS ---------------------------------------------------------
    labeldf = pd.DataFrame()
    start_time = dataset_waist['loggingTime(txt)'][1]
    start_time_since1970 = dataset_waist['accelerometerTimestamp_sinceReboot(s)'][1]

    currentstate = 0
    for index, row in dataset_waist.iterrows():
        if (currentstate != row['state(N)']):

            end_time = (row['loggingTime(txt)'])
            print('--------------------------------------------------')
            print(start_time)
            print (currentstate)
            print(end_time)
            print('--------------------------------------------------')
            if (currentstate == 0): thelabel = 'standing'
            elif (currentstate == 1): thelabel = 'indoorWalking'
            elif (currentstate == 2): thelabel = 'OutdoorWalkingStone'
            elif (currentstate == 3): thelabel = 'OutdoorWalkingGrass'
            elif (currentstate == 4): thelabel = 'OutdoorWalkingSand'
            elif (currentstate == 5): thelabel = 'onTheTable'

            labeldf = labeldf.append({'sensor_type': 'interval_label',
                                       'device_type': 'smartphone' ,
                                       'label': thelabel,
                                       'label_start_datetime': start_time,
                                       'label_end_datetime' : end_time,
                                        'state' : currentstate
                                       }, ignore_index=True)

            start_time = (row['loggingTime(txt)'])
            currentstate = row['state(N)']

    # print (labeldf)
    labeldf.to_csv(dataset_path + 'labels.csv')






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
    DataViz.plot_dataset_boxplot(dataset, ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'])

    # Plot all data

    DataViz.plot_dataset(dataset, ['acc_', 'gyr_','label'],
                         ['like', 'like','like'],
                         ['line', 'line','points'])
    # And print a summary of the dataset

    # util.print_statistics(dataset)
        # datasets.append(copy.deepcopy(dataset))

    # Finally, store the last dataset we have generated (250 ms).
    # dataset.to_csv(result_dataset_path + 'mydata_final_chapter2_result.csv')
    print ('done')




def normalityCheck():

    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter2_result.csv',index_col=0)
    dataset.index = dataset.index.to_datetime()
    i =0
    for col in [c for c in dataset.columns if not 'label' in c]:

        print 'normality for : ', col
        i = i +1

        # print(normaltest(dataset[c]))
        # print(shapiro(dataset[c]))

        ax = plt.subplot(4,3,i)

        p = dataset[col].hist()
        # p = stats.probplot(dataset[col], dist="norm", plot=pylab,fit=True)
        ax.set_title(col)
        ax.set_xlabel("")
    pylab.show()



def chauvenet():
    # And investigate the approaches for all relevant attributes.
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter2_result.csv',index_col=0)
    dataset.index = dataset.index.to_datetime()

    DataViz = VisualizeDataset()
    outlier_columns = ['gyr_ankle_x']

    for col in outlier_columns:
        # And try out all different approaches. Note that we have done some optimization
        # of the parameter values for each of the approaches by visual inspection.
        dataset = OutlierDistr.chauvenet(dataset, col)
        print("plot_binary_outliers")
        DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
        dataset = OutlierDistr.mixture_model(dataset, col)
        print("mixture_model")
        DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'])
        # This requires:
        # n_data_points * n_data_points * point_size =
        # 31839 * 31839 * 64 bits = ~8GB available memory
        try:
            dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.10, 0.99)
            print("simple_distance_based")
            DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
        except MemoryError as e:
            print('Not enough memory available for simple distance-based outlier detection...')
            print('Skipping.')

        try:
            dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
            print("local_outlier_factor")
            DataViz.plot_dataset(dataset, [col, 'lof'], ['exact', 'exact'], ['line', 'points'])
        except MemoryError as e:
            print('Not enough memory available for lof...')
            print('Skipping.')

        # Remove all the stuff from the dataset again.
        cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
        for to_remove in cols_to_remove:
            if to_remove in dataset:
                del dataset[to_remove]

    # We take Chauvent's criterion and apply it to all but the label data...

    for col in [c for c in dataset.columns if not 'label' in c]:
        print 'Measurement is now: ', col
        dataset = OutlierDistr.chauvenet(dataset, col)
        dataset.loc[dataset[col + '_outlier'] == True, col] = np.nan
        del dataset[col + '_outlier']


    dataset.to_csv(result_dataset_path + 'mydata_final_chapter3_result_outliers.csv')

def imputing():

    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter3_result_outliers.csv',index_col=0)
    dataset.index = dataset.index.to_datetime()

    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()

    #visualizing missing values
    # imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'acc_ankle_x')
    # imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'acc_ankle_x')
    # DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], 'acc_ankle_x',
    #                             imputed_mean_dataset['acc_ankle_x'], imputed_interpolation_dataset['acc_ankle_x'])


    # And we impute for all columns except for the label in the selected way (interpolation)
    for col in [c for c in dataset.columns if not 'label' in c]:
        dataset = MisVal.impute_interpolate(dataset, col)

    dataset.to_csv(result_dataset_path + 'mydata_final_chapter3_result_imputed.csv')

    # # Let us try the Kalman filter on the light_phone_lux attribute and study the result.
    #
    # original_dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter2_result.csv', index_col=0)
    # original_dataset.index = original_dataset.index.to_datetime()
    # KalFilter = KalmanFilters()
    # for col in [c for c in original_dataset.columns if not 'label' in c]:
    #     kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, col)
    #     DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], col,
    #                             kalman_dataset[col + '_kalman'])
    #     DataViz.plot_dataset(kalman_dataset, [col, col + '_kalman'], ['exact', 'exact'], ['line', 'line'])

def missingvalues():

    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter3_result_outliers.csv',index_col=0)
    dataset.index = dataset.index.to_datetime()

    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()

    # visualizing missing values
    for col in [c for c in dataset.columns if not 'label' in c]:
        imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), col)
        imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), col)
        DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], col,
                                    imputed_mean_dataset['acc_ankle_x'], imputed_interpolation_dataset['acc_ankle_x'])


def lowpass():
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter3_result_imputed.csv',index_col=0)
    dataset.index = dataset.index.to_datetime()

    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

    LowPass = LowPassFilter()

    # Computer the number of milliseconds covered by an instane based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    # Determine the sampling frequency.
    fs = float(1000) / milliseconds_per_instance


    # for cutoff in np.arange(0.1, 2.0, 0.5):
    #     # Let us study acc_phone_x:
    #     new_dataset = LowPass.low_pass_filter(copy.deepcopy(dataset), 'acc_waist_x', fs, cutoff, order=10)
    #     DataViz.plot_dataset_save(new_dataset.ix[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
    #                          ['acc_waist_x', 'acc_waist_x_lowpass'], ['exact', 'exact'], ['line', 'line'],'acc_waist_x_'+ str(int(cutoff*10)))

    cutoff = 0.5

    # And not let us include all measurements that have a form of periodicity (and filter them):
    periodic_measurements = ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                             'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                             'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                             'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                             ]
    for cutoff in np.arange(0.1, 2.0, 0.3):
        for col in periodic_measurements:
            dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
            # dataset[col] = dataset[col + '_lowpass_'+ str(int(cutoff*10))]
            # del dataset[col + '_lowpass']
    dataset.to_csv(result_dataset_path + 'mydata_final_chapter3_result_lowpass.csv')

def pca():
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter3_result_lowpass.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()

    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Determine the PC's for all but our target columns (the labels and the heart rate)
    # We simplify by ignoring both, we could also ignore one first, and apply a PC to the remainder.

    PCA = PrincipalComponentAnalysis()
    selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c))]
    pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

    # Plot the variance explained.

    plt.plot(range(1, len(selected_predictor_cols) + 1), pc_values, 'b-')
    plt.xlabel('principal component number')
    plt.ylabel('explained variance')
    # plt.show(block=False)
    plt.show()


    # We select 7 as the best number of PC's as this explains most of the variance
    # exit(0)
    n_pcs = 20

    dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

    # And we visualize the result of the PC's

    DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])

    # And the overall final dataset:

    DataViz.plot_dataset(dataset,
                         ['acc_', 'gyr_', 'lowpass_', 'pca_', 'label'],
                         [ 'like', 'like','like',  'like', 'like'],
                         [ 'line', 'line', 'line','points', 'points'])

    # Store the outcome.

    dataset.to_csv(result_dataset_path + 'mydata_final_chapter3_result_pca.csv')

def timedomain():
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter3_result_pca.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # First we focus on the time domain.
    print("doing time domain")
    # Set the window sizes to the number of instances representing  5 seconds, 30 seconds and 5 minutes
    # window_sizes = [int(float(5000) / milliseconds_per_instance), int(float(0.5 * 60000) / milliseconds_per_instance),
    #                 int(float(5 * 60000) / milliseconds_per_instance)]
    #
    NumAbs = NumericalAbstraction()
    # dataset_copy = copy.deepcopy(dataset)
    # for ws in window_sizes:
    #     dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_waist_x'], ws, 'mean')
    #     dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_waist_x'], ws, 'std')
    #
    # DataViz.plot_dataset(dataset_copy, ['acc_waist_x', 'acc_waist_x_temp_mean', 'acc_waist_x_temp_std', 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
    # exit(0)
    ws = int(float(5000) / milliseconds_per_instance)
    selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
    dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
    dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

    CatAbs = CategoricalAbstraction()
    dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03,
                                          int(float(5000) / milliseconds_per_instance), 2)
    print("finished time domain")

    dataset.to_csv(result_dataset_path + 'mydata_final_chapter4_result_timedomain.csv')

def frequencydomain():
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter4_result_timedomain.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Now we move to the frequency domain, with the same window size.
    print("doing freq domain")
    FreqAbs = FourierTransformation()
    fs = float(1000) / milliseconds_per_instance
    ws = int(float(5000) / milliseconds_per_instance)


    # data_table = FreqAbs.abstract_frequency(copy.deepcopy(dataset), ['acc_waist_x'],int(float(10000) / milliseconds_per_instance), fs)

    # Spectral analysis.

    # DataViz.plot_dataset(data_table, ['acc_waist_x_max_freq', 'acc_waist_x_freq_weighted', 'acc_waist_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])

    # exit(0)
    periodic_predictor_cols = ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                               'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                               'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                               'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                               ]

    dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols,
                                         int(float(10000) / milliseconds_per_instance), fs)

    # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

    # The percentage of overlap we allow
    window_overlap = 0.9
    skip_points = int((1 - window_overlap) * ws)
    dataset = dataset.iloc[::skip_points, :]

    dataset.to_csv(result_dataset_path + 'mydata_final_chapter4_result_frequencydomain.csv')
    print("finished frq domain")

def clustering():
    print("doing clustering")
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter4_result_frequencydomain.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()

    # # First let us use non hierarchical clustering.
    #
    clusteringNH = NonHierarchicalClustering()

    # Let us look at k-means first.

    # k_values = range(2, 10)
    # silhouette_values = []
    # #
    # ## Do some initial runs to determine the right number for k
    # #
    # print '===== kmeans clustering ====='
    # for k in k_values:
    #     print 'k = ', k
    #     dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], k, 'default', 50, 50)
    #     silhouette_score = dataset_cluster['silhouette'].mean()
    #     print 'silhouette = ', silhouette_score
    #     silhouette_values.append(silhouette_score)
    #
    # plt.plot(k_values, silhouette_values, 'b-')
    # plt.xlabel('k')
    # plt.ylabel('silhouette score')
    # plt.ylim([0,1])
    # plt.show()
    #
    # # # And run the knn with the highest silhouette score
    # # exit(0)
    # k = 3
    # # silhouette = 0.771288241913
    # #
    # dataset_knn = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], k, 'default', 50, 50)
    # DataViz.plot_clusters_3d(dataset_knn, ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], 'cluster', ['label'])
    # DataViz.plot_silhouette(dataset_knn, 'cluster', 'silhouette')
    # util.print_latex_statistics_clusters(dataset_knn, 'cluster', ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], 'label')
    # del dataset_knn['silhouette']

    #
    # k_values = range(2, 10)
    # silhouette_values = []
    #
    # # Do some initial runs to determine the right number for k
    #
    # print '===== k medoids clustering ====='
    # for k in k_values:
    #     print 'k = ', k
    #     dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], k, 'default', 20, n_inits=10)
    #     silhouette_score = dataset_cluster['silhouette'].mean()
    #     print 'silhouette = ', silhouette_score
    #     silhouette_values.append(silhouette_score)
    #
    # plt.plot(k_values, silhouette_values, 'b-')
    # plt.ylim([0,1])
    # plt.xlabel('k')
    # plt.ylabel('silhouette score')
    # plt.show()

    # And run k medoids with the highest silhouette score

    # k = 3

    # dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], k, 'default', 20, n_inits=10)
    # DataViz.plot_clusters_3d(dataset_kmed, ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], 'cluster', ['label'])
    # DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
    # util.print_latex_statistics_clusters(dataset_kmed, 'cluster', ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], 'label')

    # And the hierarchical clustering is the last one we try

    clusteringH = HierarchicalClustering()

    k_values = range(2, 10)
    silhouette_values = []

    # Do some initial runs to determine the right number for the maximum number of clusters.

    # print '===== agglomaritive clustering ====='
    # for k in k_values:
    #     print 'k = ', k
    #     dataset_cluster, l = clusteringH.agglomerative_over_instances(copy.deepcopy(dataset),
    #                                                                   ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'], 5,
    #                                                                   'euclidean', use_prev_linkage=True,
    #                                                                   link_function='ward')
    #     print (l)
    #     silhouette_score = dataset_cluster['silhouette'].mean()
    #     print 'silhouette = ', silhouette_score
    #     silhouette_values.append(silhouette_score)
    #     if k == k_values[0]:
    #         DataViz.plot_dendrogram(dataset_cluster, l)
    #
    # plt.plot(k_values, silhouette_values, 'b-')
    # plt.ylim([0, 1])
    # plt.xlabel('max number of clusters')
    # plt.ylabel('silhouette score')
    # plt.show()
    # And we select the outcome dataset of the knn clustering....



    # util.print_latex_statistics_clusters(dataset_cluster, 'cluster', ['acc_waist_x', 'acc_waist_y', 'acc_waist_z'],
    #                                      'label')
    # del dataset_kmed['silhouette']
    # dataset_kmed.to_csv(result_dataset_path + 'mydata_final_chapter5_result.csv')
    print("finished clustering")


def forward():
    print("doing forward selection")
    # dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter5_result.csv', index_col=0)
    # dataset.index = dataset.index.to_datetime()
    # milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    # MisVal = ImputationMissingValues()
    # DataViz = VisualizeDataset()
    # # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.
    #
    # # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # # cases where we do not know the label.
    #
    # prepare = PrepareDatasetForLearning()
    #
    # # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)
    # #train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)
    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification_temporal(dataset, ['label'], 'like',
    #                                                                                         0.5,
    #                                                                                         filter=True)
    #
    #
    # print 'Training set length is: ', len(train_X.index)
    # print 'Test set length is: ', len(test_X.index)
    #
    # # Select subsets of the features that we will consider:
    #
    # basic_features =  ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
    #                            'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
    #                            'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
    #                            'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
    #                            ]
    # pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10',
    #                 'pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20']
    # time_features = [name for name in dataset.columns if '_temp_' in name]
    # freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    # print '#basic features: ', len(basic_features)
    # print '#PCA features: ', len(pca_features)
    # print '#time features: ', len(time_features)
    # print '#frequency features: ', len(freq_features)
    # cluster_features = ['cluster']
    # print '#cluster features: ', len(cluster_features)
    # features_after_chapter_3 = list(set().union(basic_features, pca_features))
    # features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    # features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))
    #
    #
    # # First, let us consider the performance over a selection of features:
    #
    # fs = FeatureSelectionClassification()

    # features, ordered_features, ordered_scores = fs.forward_selection(10, train_X[features_after_chapter_5], train_y)
    # features, ordered_features, ordered_scores = fs.backward_selection(10, train_X[features_after_chapter_5], train_y)
    # ordered_scores = [0.58258135083401696, 0.88843314191960621, 0.93669674596663932, 0.96431501230516814, 0.977167076838939,
    #  0.98304621274268522, 0.98646431501230514, 0.98837845228329235, 0.98947224500957065, 0.99042931364506426]
    # ordered_features = ['acc_ankle_z_temp_std_ws_20', 'acc_waist_z_freq_0.0_Hz_ws_40', 'lowpass_gyr_waist_x_1_temp_mean_ws_20',
    #  'lowpass_gyr_ankle_x_1_temp_mean_ws_20', 'lowpass_acc_ankle_x_1_temp_mean_ws_20',
    #  'lowpass_acc_ankle_y_1_temp_mean_ws_20', 'lowpass_gyr_waist_y_1_temp_mean_ws_20',
    #  'lowpass_gyr_ankle_y_1_temp_mean_ws_20', 'lowpass_acc_ankle_x_13_temp_std_ws_20',
    #  'lowpass_acc_ankle_z_1_temp_std_ws_20']


    ordered_scores= [0.55434366628396481, 0.654228855721393, 0.84041331802525832, 0.9783773440489858, 0.98603138155376957,
     0.98985840030616146, 0.99119785686949868, 0.99311136624569463, 0.9940681209337926, 0.99502487562189057]
    ordered_features = ['lowpass_gyr_ankle_x_1_temp_mean_ws_20', 'acc_waist_z_freq_0.0_Hz_ws_40', 'acc_ankle_x_freq_0.0_Hz_ws_40',
     'lowpass_acc_ankle_z_19_temp_std_ws_20', 'lowpass_acc_ankle_z_1_temp_mean_ws_20',
     'lowpass_gyr_ankle_y_7_temp_std_ws_20', 'acc_waist_y_temp_std_ws_20', 'pca_15_temp_mean_ws_20',
     'lowpass_acc_ankle_x_10_temp_mean_ws_20', 'pca_7_temp_mean_ws_20']

    print ordered_scores
    print ordered_features



    plt.plot(range(1, 11), ordered_scores)
    plt.xlabel('number of features')
    plt.ylabel('accuracy')
    plt.show()

def gridsearch():
    print("doing grid search")
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter5_result.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    prepare = PrepareDatasetForLearning()

    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.5, filter=True, temporal=True)
    #train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)
    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification_temporal(dataset, ['label'], 'like',
                                                                                            0.5,
                                                                                            filter=True)


    print 'Training set length is: ', len(train_X.index)
    print 'Test set length is: ', len(test_X.index)

    # Select subsets of the features that we will consider:

    basic_features =  ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                               'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                               'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                               'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                               ]
    pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10',
                    'pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print '#basic features: ', len(basic_features)
    print '#PCA features: ', len(pca_features)
    print '#time features: ', len(time_features)
    print '#frequency features: ', len(freq_features)
    cluster_features = ['cluster']
    print '#cluster features: ', len(cluster_features)
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    # possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    # feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5]
    feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5']

    repeats = 1

    scores_over_all_algs = []

    for i in range(0, len(possible_feature_sets)):
        selected_train_X = train_X[possible_feature_sets[i]]
        selected_test_X = test_X[possible_feature_sets[i]]

        # First we run our non deterministic classifiers a number of times to average their score.

        performance_tr_nn = 0
        performance_tr_rf = 0
        performance_tr_svm = 0
        performance_te_nn = 0
        performance_te_rf = 0
        performance_te_svm = 0

        for repeat in range(0, repeats):
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(selected_train_X, train_y, selected_test_X, gridsearch=True)
            performance_tr_nn += eval.accuracy(train_y, class_train_y)
            performance_te_nn += eval.accuracy(test_y, class_test_y)

            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(selected_train_X, train_y, selected_test_X, gridsearch=True)
            performance_tr_rf += eval.accuracy(train_y, class_train_y)
            performance_te_rf += eval.accuracy(test_y, class_test_y)

            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(selected_train_X, train_y, selected_test_X, gridsearch=True)
            performance_tr_svm += eval.accuracy(train_y, class_train_y)
            performance_te_svm += eval.accuracy(test_y, class_test_y)


        overall_performance_tr_nn = performance_tr_nn/repeats
        overall_performance_te_nn = performance_te_nn/repeats
        overall_performance_tr_rf = performance_tr_rf/repeats
        overall_performance_te_rf = performance_te_rf/repeats
        overall_performance_tr_svm = performance_tr_svm/repeats
        overall_performance_te_svm = performance_te_svm/repeats

        # And we run our deterministic classifiers:


        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(selected_train_X, train_y, selected_test_X, gridsearch=True)
        performance_tr_knn = eval.accuracy(train_y, class_train_y)
        performance_te_knn = eval.accuracy(test_y, class_test_y)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(selected_train_X, train_y, selected_test_X, gridsearch=True)
        performance_tr_dt = eval.accuracy(train_y, class_train_y)
        performance_te_dt = eval.accuracy(test_y, class_test_y)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(selected_train_X, train_y, selected_test_X)
        performance_tr_nb = eval.accuracy(train_y, class_train_y)
        performance_te_nb = eval.accuracy(test_y, class_test_y)

        scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                    (overall_performance_tr_nn, overall_performance_te_nn),
                                                                                                    (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                    (overall_performance_tr_svm, overall_performance_te_svm),
                                                                                                    (performance_tr_knn, performance_te_knn),
                                                                                                    (performance_tr_dt, performance_te_dt),
                                                                                                    (performance_tr_nb, performance_te_nb)])
        scores_over_all_algs.append(scores_with_sd)

    DataViz.plot_performances_classification(['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)

def randomforest():
    print("doing randomforest")
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter5_result.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    prepare = PrepareDatasetForLearning()

    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)
    #train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)
    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification_temporal(dataset, ['label'], 'like',
                                                                                            0.5,
                                                                                            filter=True)


    print 'Training set length is: ', len(train_X.index)
    print 'Test set length is: ', len(test_X.index)

    # Select subsets of the features that we will consider:

    basic_features =  ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                               'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                               'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                               'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                               ]
    pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10',
                    'pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print '#basic features: ', len(basic_features)
    print '#PCA features: ', len(pca_features)
    print '#time features: ', len(time_features)
    print '#frequency features: ', len(freq_features)
    cluster_features = ['cluster']
    print '#cluster features: ', len(cluster_features)
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    selected_features = ['lowpass_gyr_ankle_x_1_temp_mean_ws_20', 'acc_waist_z_freq_0.0_Hz_ws_40',
                        'acc_ankle_x_freq_0.0_Hz_ws_40',
                        'lowpass_acc_ankle_z_19_temp_std_ws_20', 'lowpass_acc_ankle_z_1_temp_mean_ws_20',
                        'lowpass_gyr_ankle_y_7_temp_std_ws_20', 'acc_waist_y_temp_std_ws_20', 'pca_15_temp_mean_ws_20',
                        'lowpass_acc_ankle_x_10_temp_mean_ws_20', 'pca_7_temp_mean_ws_20']
    # possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    # feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    possible_feature_sets = [basic_features,features_after_chapter_5]
    feature_names = ['initial set', 'Chapter 5']

    # class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5],
    #                                                                                            gridsearch=True,
    #                                                                                            print_model_details=True, export_tree_path=result_dataset_path)
    #

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5],
                                                                                               gridsearch=True, print_model_details=True)


    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)


def svm():
    print("doing svm")
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter5_result.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    prepare = PrepareDatasetForLearning()


    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.5,
                                                                                   filter=True, temporal=False)
    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)

    print 'Training set length is: ', len(train_X.index)
    print 'Test set length is: ', len(test_X.index)

    # Select subsets of the features that we will consider:

    basic_features = ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                      'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                      'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                      'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                      ]
    pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10',
                    'pca_11', 'pca_12', 'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pca_17', 'pca_18', 'pca_19', 'pca_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print '#basic features: ', len(basic_features)
    print '#PCA features: ', len(pca_features)
    print '#time features: ', len(time_features)
    print '#frequency features: ', len(freq_features)
    cluster_features = ['cluster']
    print '#cluster features: ', len(cluster_features)
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(
        set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    # possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    # feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    possible_feature_sets = [basic_features, features_after_chapter_5]
    feature_names = ['initial set', 'Chapter 5']

    # {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5],  kernel = 'rbf', C = 100, gamma =  0.0001,gridsearch=False,print_model_details=True)
    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)

def temporalsplitsvm():
    print("doing svm")
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter5_result.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    prepare = PrepareDatasetForLearning()


    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification_temporal(dataset, ['label'], 'like', 0.5,
                                                                                   filter=True)

    # print((train_y))
    #
    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like',
    #                                                                                         0.5,
    #                                                                                         filter=True,temporal=False)
    #
    # print((train_y))
    #
    # exit(0)
    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)

    print 'Training set length is: ', len(train_X.index)
    print 'Test set length is: ', len(test_X.index)

    # Select subsets of the features that we will consider:

    basic_features = ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                      'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                      'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                      'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                      ]
    pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10',
                    'pca_11', 'pca_12', 'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pca_17', 'pca_18', 'pca_19', 'pca_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print '#basic features: ', len(basic_features)
    print '#PCA features: ', len(pca_features)
    print '#time features: ', len(time_features)
    print '#frequency features: ', len(freq_features)
    cluster_features = ['cluster']
    print '#cluster features: ', len(cluster_features)
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(
        set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    # possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    # feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    possible_feature_sets = [basic_features, features_after_chapter_5]
    feature_names = ['initial set', 'Chapter 5']

    # {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5],  gridsearch=True,print_model_details=True)
    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)


def temporalsplitnn():
    print("doing feedforward_neural_network")
    dataset = pd.read_csv(result_dataset_path + 'mydata_final_chapter5_result.csv', index_col=0)
    dataset.index = dataset.index.to_datetime()
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000
    MisVal = ImputationMissingValues()
    DataViz = VisualizeDataset()
    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    prepare = PrepareDatasetForLearning()


    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification_temporal(dataset, ['label'], 'like', 0.5,
                                                                                   filter=True)

    # print((train_y))
    #
    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like',
    #                                                                                         0.5,
    #                                                                                         filter=True,temporal=False)
    #
    # print((train_y))
    #
    # exit(0)
    # train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)

    print 'Training set length is: ', len(train_X.index)
    print 'Test set length is: ', len(test_X.index)

    # Select subsets of the features that we will consider:

    basic_features = ['acc_waist_x', 'acc_waist_y', 'acc_waist_z',
                      'gyr_waist_x', 'gyr_waist_y', 'gyr_waist_z',
                      'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
                      'gyr_ankle_x', 'gyr_ankle_y', 'gyr_ankle_z'
                      ]
    pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10',
                    'pca_11', 'pca_12', 'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pca_17', 'pca_18', 'pca_19', 'pca_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print '#basic features: ', len(basic_features)
    print '#PCA features: ', len(pca_features)
    print '#time features: ', len(time_features)
    print '#frequency features: ', len(freq_features)
    cluster_features = ['cluster']
    print '#cluster features: ', len(cluster_features)
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(
        set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    # possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    # feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    possible_feature_sets = [basic_features, features_after_chapter_5]
    feature_names = ['initial set', 'Chapter 5']

    # {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5] ,gridsearch=True,print_model_details=True)
    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)


# makeDataFile()
# makeLabelFile()
# makeDataSet(250)
# makeDataSet(100)
# makeDataSet(1000)
# makeDataSet(10000)
normalityCheck()
# missingvalues()
# chauvenet()
# imputing()

# lowpass()
# pca()
# timedomain()
# frequencydomain()
# clustering()

# forward()
# gridsearch()

# randomforest()
# svm()
# temporalsplitsvm()
# temporalsplitnn()
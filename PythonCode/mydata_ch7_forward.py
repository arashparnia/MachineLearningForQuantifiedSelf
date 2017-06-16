##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

import os

import matplotlib.pyplot as plot
import numpy
import pandas as pd

from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from util import util
from util.VisualizeDataset import VisualizeDataset

# # Of course we repeat some stuff from Chapter 3, namely to load the dataset
#
# DataViz = VisualizeDataset()
#
# # Read the result from the previous chapter, and make sure the index is of the type datetime.
#
# dataset_path = './intermediate_datafiles/'
# export_tree_path = 'Example_graphs/Chapter7/'
#
# try:
#     dataset = pd.read_csv(dataset_path + 'mydata_chapter5_result.csv', index_col=0)
# except IOError as e:
#     print('File not found, try to run previous crowdsignals scripts first!')
#     raise e
#
# if not os.path.exists(export_tree_path):
#     os.makedirs(export_tree_path)
#
# dataset.index = dataset.index.to_datetime()
#
# # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.
#
# # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# # cases where we do not know the label.
#
# prepare = PrepareDatasetForLearning()
#
# train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)
# #train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)
#
# print 'Training set length is: ', len(train_X.index)
# print 'Test set length is: ', len(test_X.index)
#
# # Select subsets of the features that we will consider:
#
# basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','mag_phone_x','mag_phone_y','mag_phone_z','press_phone_Pressure']
# pca_features = ['pca_1','pca_2','pca_3','pca_4']
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
#
# features, ordered_features, ordered_scores = fs.forward_selection(50, train_X[features_after_chapter_5], train_y)
# # features, ordered_features, ordered_scores = fs.backward_selection(10, train_X[features_after_chapter_5], train_y)
# print ordered_scores
# print ordered_features

ordered_scores = [0.81599999999999995, 0.96799999999999997, 0.98560000000000003, 0.99119999999999997, 0.99439999999999995, 0.99519999999999997, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002, 0.99680000000000002]
ordered_features = ['pca_1_temp_mean_ws_120', 'press_phone_Pressure_temp_mean_ws_120', 'acc_phone_x_freq_0.0_Hz_ws_40', 'mag_phone_y_temp_mean_ws_120', 'gyr_phone_z_freq_0.7_Hz_ws_40', 'mag_phone_z_freq_0.9_Hz_ws_40', 'gyr_phone_y_freq_weighted', 'acc_phone_z_freq_0.2_Hz_ws_40', 'acc_phone_y_freq_0.5_Hz_ws_40', 'mag_phone_y_freq_0.4_Hz_ws_40', 'gyr_phone_z_freq_0.5_Hz_ws_40', 'mag_phone_z_freq_1.3_Hz_ws_40', 'gyr_phone_z_freq_0.2_Hz_ws_40', 'acc_phone_y_freq_0.9_Hz_ws_40', 'acc_phone_z_freq_1.6_Hz_ws_40', 'acc_phone_x_freq_0.3_Hz_ws_40', 'gyr_phone_z_freq_1.9_Hz_ws_40', 'mag_phone_y_freq_0.1_Hz_ws_40', 'mag_phone_y_freq_0.0_Hz_ws_40', 'acc_phone_z_freq_0.6_Hz_ws_40', 'gyr_phone_y_max_freq', 'mag_phone_z_freq_0.6_Hz_ws_40', 'gyr_phone_x_max_freq', 'gyr_phone_x_freq_0.5_Hz_ws_40', 'gyr_phone_y_freq_0.6_Hz_ws_40', 'gyr_phone_x_freq_weighted', 'acc_phone_y_freq_0.3_Hz_ws_40', 'gyr_phone_z_freq_0.1_Hz_ws_40', 'acc_phone_z_freq_0.9_Hz_ws_40', 'mag_phone_y_freq_0.2_Hz_ws_40', 'gyr_phone_y_freq_1.1_Hz_ws_40', 'mag_phone_z_freq_1.5_Hz_ws_40', 'gyr_phone_y_freq_2.0_Hz_ws_40', 'gyr_phone_z_freq_1.6_Hz_ws_40', 'acc_phone_y_freq_0.2_Hz_ws_40', 'acc_phone_x_freq_1.3_Hz_ws_40', 'acc_phone_y_freq_1.1_Hz_ws_40', 'acc_phone_x_freq_weighted', 'acc_phone_z_freq_0.7_Hz_ws_40', 'gyr_phone_y_freq_1.3_Hz_ws_40', 'gyr_phone_x_freq_0.8_Hz_ws_40', 'acc_phone_x_freq_1.9_Hz_ws_40', 'mag_phone_z_freq_0.4_Hz_ws_40', 'mag_phone_x_freq_0.1_Hz_ws_40', 'acc_phone_y_freq_0.4_Hz_ws_40', 'acc_phone_y_freq_weighted', 'mag_phone_z_freq_0.5_Hz_ws_40', 'mag_phone_y_freq_1.3_Hz_ws_40', 'gyr_phone_x_freq_0.3_Hz_ws_40', 'mag_phone_x_freq_0.5_Hz_ws_40']

plot.plot(range(1, 51), ordered_scores)
plot.xlabel('number of features')
plot.ylabel('accuracy')
plot.show()

# Based on the plot we select the top 10 featu res.
exit(0)

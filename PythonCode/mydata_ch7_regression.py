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

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.

dataset_path = './intermediate_datafiles/'
export_tree_path = 'Example_graphs/Chapter7/'

try:
    dataset = pd.read_csv(dataset_path + 'mydata_chapter5_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

if not os.path.exists(export_tree_path):
    os.makedirs(export_tree_path)

dataset.index = dataset.index.to_datetime()

# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)
#train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.01, filter=True, temporal=False)

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','mag_phone_x','mag_phone_y','mag_phone_z','press_phone_Pressure']
pca_features = ['pca_1','pca_2','pca_3','pca_4']
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


# First, let us consider the performance over a selection of features:

fs = FeatureSelectionClassification()

features, ordered_features, ordered_scores = fs.forward_selection(50, train_X[features_after_chapter_5], train_y)
# features, ordered_features, ordered_scores = fs.backward_selection(10, train_X[features_after_chapter_5], train_y)
print ordered_scores
print ordered_features

plot.plot(range(1, 11), ordered_scores)
plot.xlabel('number of features')
plot.ylabel('accuracy')
plot.show()

# Based on the plot we select the top 10 features.
exit(0)
# selected_features = ['acc_phone_y_freq_0.0_Hz_ws_40', 'press_phone_pressure_temp_mean_ws_120', 'gyr_phone_x_temp_std_ws_120',
#                      'mag_watch_y_pse', 'mag_phone_z_max_freq', 'gyr_watch_y_freq_weighted', 'gyr_phone_y_freq_1.0_Hz_ws_40',
#                      'acc_phone_x_freq_1.9_Hz_ws_40', 'mag_watch_z_freq_0.9_Hz_ws_40', 'acc_watch_y_freq_0.5_Hz_ws_40']
selected_features =  ['pca_1_temp_mean_ws_120', 'press_phone_Pressure_temp_mean_ws_120', 'acc_phone_x_freq_0.0_Hz_ws_40', 'mag_phone_y_temp_mean_ws_120', 'gyr_phone_z_freq_0.7_Hz_ws_40', 'pca_3', 'mag_phone_z_freq_0.9_Hz_ws_40', 'mag_phone_z_freq_0.7_Hz_ws_40', 'gyr_phone_z_freq_0.5_Hz_ws_40', 'gyr_phone_y_freq_0.6_Hz_ws_40'] # forward feature selection



# # Let us first study the impact of regularization and model complexity: does regularization prevent overfitting?
#
learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()
#
# reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
# performance_training = []
# performance_test = []
#
# # We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.
#
# repeats = 2
#
# for reg_param in reg_parameters:
#     performance_tr = 0
#     performance_te = 0
#     for i in range(0, repeats):
#         class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(train_X, train_y,
#                                                                                                             test_X, hidden_layer_sizes=(250, ), alpha=reg_param, max_iter=500,
#                                                                                                             gridsearch=False)
#         performance_tr += eval.accuracy(train_y, class_train_y)
#         performance_te += eval.accuracy(test_y, class_test_y)
#     performance_training.append(performance_tr/repeats)
#     performance_test.append(performance_te/repeats)
#
# plot.hold(True)
# plot.semilogx(reg_parameters, performance_training, 'r-')
# plot.semilogx(reg_parameters, performance_test, 'b:')
# print performance_training
# print performance_test
# plot.xlabel('regularization parameter value')
# plot.ylabel('accuracy')
# plot.ylim([0.95, 1.01])
# plot.legend(['training', 'test'], loc=4)
# plot.hold(False)
#
# plot.show()

# Second, let us consider the influence of certain parameter settings (very related to the regulariztion) and study the impact on performance.
# leaf_settings = numpy.arange(1, 50, 10)
# leaf_settings = [1,2,5,10,20,25]
# performance_training = []
# performance_test = []
#
# for no_points_leaf in leaf_settings:
#     class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[selected_features], train_y, test_X[selected_features], min_samples_leaf=no_points_leaf,
#                                                                                                gridsearch=False, print_model_details=True)
#     performance_training.append(eval.accuracy(train_y, class_train_y))
#     performance_test.append(eval.accuracy(test_y, class_test_y))
#
# plot.hold(True)
# plot.plot(leaf_settings, performance_training, 'r-')
# plot.plot(leaf_settings, performance_test, 'b:')
# plot.xlabel('minimum number of points per leaf')
# plot.ylabel('accuracy')
# plot.legend(['training', 'test'], loc=1)
# plot.hold(False)
#
# plot.show()


# So yes, it is important :) Therefore we perform grid searches over the most important parameters, and do so by means
# of cross validation upon the training set.


possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']
repeats = 5

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

# And we study two promising ones in more detail. First let us consider the decision tree which works best with the selected
# features.
#
class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[selected_features], train_y, test_X[selected_features],
                                                                                           gridsearch=True,
                                                                                           print_model_details=True, export_tree_path=export_tree_path)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(train_X[selected_features], train_y, test_X[selected_features],
                                                                                           gridsearch=True, print_model_details=True)

test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)

exit(0)
#
# ##############################################################
# #                                                            #
# #    Mark Hoogendoorn and Burkhardt Funk (2017)              #
# #    Machine Learning for the Quantified Self                #
# #    Springer                                                #
# #    Chapter 7                                               #
# #                                                            #
# ##############################################################
#
# import os
#
# import pandas as pd
#
# from Chapter7.Evaluation import RegressionEvaluation
# from Chapter7.FeatureSelection import FeatureSelectionRegression
# from Chapter7.LearningAlgorithms import RegressionAlgorithms
# from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
# from util import util
# from util.VisualizeDataset import VisualizeDataset
#
# # Of course we repeat some stuff from Chapter 3, namely to load the dataset
#
# DataViz = VisualizeDataset()
#
# # Read the result from the previous chapter, and make sure the index is of the type datetime.
# dataset_path = './intermediate_datafiles/'
# export_tree_path = 'Example_graphs/Chapter7/'
#
# dataset = pd.read_csv(dataset_path + 'chapter5_result.csv', index_col=0)
# dataset.index = dataset.index.to_datetime()
#
# if not os.path.exists(export_tree_path):
#     os.makedirs(export_tree_path)
#
# # Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.
#
# prepare = PrepareDatasetForLearning()
#
# train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate', '2016-02-08 18:28:56',
#                                                                                    '2016-02-08 19:34:07', '2016-02-08 20:07:50')
# #                                                                                   '2016-02-08 18:28:58','2016-02-08 18:28:59')
#
# print 'Training set length is: ', len(train_X.index)
# print 'Test set length is: ', len(test_X.index)
#
# # Select subsets of the features that we will consider:
#
# basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','gyr_watch_x','gyr_watch_y','gyr_watch_z',
#                   'labelOnTable','labelSitting','labelWashingHands','labelWalking','labelStanding','labelDriving','labelEating','labelRunning',
#                   'light_phone_lux','mag_phone_x','mag_phone_y','mag_phone_z','mag_watch_x','mag_watch_y','mag_watch_z','press_phone_pressure']
# pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
# time_features = [name for name in dataset.columns if ('temp_' in name and not 'hr_watch' in name)]
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
# fs = FeatureSelectionRegression()
#
# # First, let us consider the Pearson correlations and see whether we can select based on them.
# features, correlations = fs.pearson_selection(10, train_X[features_after_chapter_5], train_y)
# util.print_pearson_correlations(correlations)
#
# # We select the 10 features with the highest correlation.
#
# selected_features = ['temp_pattern_labelOnTable','labelOnTable','temp_pattern_labelOnTable(b)labelOnTable','pca_2_temp_mean_ws_120',
#                      'pca_1_temp_mean_ws_120','acc_watch_y_temp_mean_ws_120','pca_2','acc_phone_z_temp_mean_ws_120',
#                      'gyr_watch_y_pse','gyr_watch_x_pse']
#
# possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
# feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']
#
# # Let us first study the importance of the parameter settings.
#
# learner = RegressionAlgorithms()
# eval = RegressionEvaluation()
#
# # We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.
#
# repeats = 5
#
# scores_over_all_algs = []
#
# for i in range(0, len(possible_feature_sets)):
#
#     selected_train_X = train_X[possible_feature_sets[i]]
#     selected_test_X = test_X[possible_feature_sets[i]]
#
#     # First we run our non deterministic classifiers a number of times to average their score.
#
#     performance_tr_nn = 0
#     performance_tr_nn_std = 0
#     performance_tr_rf = 0
#     performance_tr_rf_std = 0
#     performance_tr_svm = 0
#     performance_tr_svm_std = 0
#     performance_te_nn = 0
#     performance_te_nn_std = 0
#     performance_te_rf = 0
#     performance_te_rf_std = 0
#     performance_te_svm = 0
#     performance_te_svm_std = 0
#
#     for repeat in range(0, repeats):
#         regr_train_y, regr_test_y = learner.feedforward_neural_network(selected_train_X, train_y, selected_test_X, gridsearch=True)
#
#         mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
#         mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
#         mean_training = eval.mean_squared_error(train_y, regr_train_y)
#         performance_tr_nn += mean_tr
#         performance_tr_nn_std += std_tr
#         performance_te_nn += mean_te
#         performance_te_nn_std += std_te
#
#         regr_train_y, regr_test_y = learner.random_forest(selected_train_X, train_y, selected_test_X, gridsearch=True)
#         mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
#         mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
#         performance_tr_rf += mean_tr
#         performance_tr_rf_std += std_tr
#         performance_te_rf += mean_te
#         performance_te_rf_std += std_te
#
#     overall_performance_tr_nn = performance_tr_nn/repeats
#     overall_performance_tr_nn_std = performance_tr_nn_std/repeats
#     overall_performance_te_nn = performance_te_nn/repeats
#     overall_performance_te_nn_std = performance_te_nn_std/repeats
#     overall_performance_tr_rf = performance_tr_rf/repeats
#     overall_performance_tr_rf_std = performance_tr_rf_std/repeats
#     overall_performance_te_rf = performance_te_rf/repeats
#     overall_performance_te_rf_std = performance_te_rf_std/repeats
#
#     # And we run our deterministic algorithms:
#
#     regr_train_y, regr_test_y = learner.support_vector_regression_without_kernel(selected_train_X, train_y, selected_test_X, gridsearch=True)
#     mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
#     mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
#     performance_tr_svm = mean_tr
#     performance_tr_svm_std = std_tr
#     performance_te_svm = mean_te
#     performance_te_svm_std = std_te
#
#     regr_train_y, regr_test_y = learner.k_nearest_neighbor(selected_train_X, train_y, selected_test_X, gridsearch=True)
#     mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
#     mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
#     performance_tr_knn = mean_tr
#     performance_tr_knn_std = std_tr
#     performance_te_knn = mean_te
#     performance_te_knn_std = std_te
#
#     regr_train_y, regr_test_y = learner.decision_tree(selected_train_X, train_y, selected_test_X, gridsearch=True, export_tree_path=export_tree_path)
#
#     mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
#     mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
#     performance_tr_dt = mean_tr
#     performance_tr_dt_std = std_tr
#     performance_te_dt = mean_te
#     performance_te_dt_std = std_te
#
#     scores_with_sd = [(overall_performance_tr_nn, overall_performance_tr_nn_std, overall_performance_te_nn, overall_performance_te_nn_std),
#                       (overall_performance_tr_rf, overall_performance_tr_rf_std, overall_performance_te_rf, overall_performance_te_rf_std),
#                       (performance_tr_svm, performance_tr_svm_std, performance_te_svm, performance_te_svm_std),
#                       (performance_tr_knn, performance_tr_knn_std, performance_te_knn, performance_te_knn_std),
#                       (performance_tr_dt, performance_tr_dt_std, performance_te_dt, performance_te_dt_std)]
#     util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), scores_with_sd)
#     scores_over_all_algs.append(scores_with_sd)
#
# print scores_over_all_algs
# DataViz.plot_performances_regression(['NN', 'RF', 'SVM', 'KNN', 'DT'], feature_names, scores_over_all_algs)
#
# regr_train_y, regr_test_y = learner.random_forest(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5], gridsearch=True, print_model_details=True)
# DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y, test_X.index, test_y, regr_test_y, 'heart rate')
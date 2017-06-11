import pandas as pd
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
# import knnimpute
from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# Let is create our visualization class again.
DataViz = VisualizeDataset()


dataset_path = './intermediate_datafiles/'
dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
print(list(dataset))
dataset = dataset[['hr_watch_rate','acc_watch_x','acc_watch_y','acc_watch_y','acc_phone_x','acc_phone_y','acc_phone_z'] ]
print(list(dataset))
dataset.index = dataset.index.to_datetime()

# Computer the number of milliseconds covered by an instane based on the first two rows
# milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

# Step 2: Let us impute the missing values.

# MisVal = ImputationMissingValues()
# imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'hr_watch_rate')
# imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'hr_watch_rate')
# imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'hr_watch_rate')
# DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], 'hr_watch_rate', imputed_mean_dataset['hr_watch_rate'], imputed_interpolation_dataset['hr_watch_rate'])






X_incomplete = dataset
# print(list(X_incomplete))
# # X is the complete data matrix
# # X_incomplete has the same values as X except a subset have been replace with NaN
#
# # Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=6).complete(X_incomplete)
# X_filled_knn = knnimpute.(X_incomplete)

DataViz.plot_imputed_values(dataset, ['original', 'imputed'], 'hr_watch_rate', X_filled_knn[:,0])


# # matrix completion using convex optimization to find low-rank solution
# # that still matches observed values. Slow!
# X_filled_nnm = NuclearNormMinimization().complete(X_incomplete)
#
# # Instead of solving the nuclear norm objective directly, instead
# # induce sparsity using singular value thresholding
# X_filled_softimpute = SoftImpute().complete(X_incomplete_normalized)
#
# # print mean squared error for the three imputation methods above
# nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()
# print("Nuclear norm minimization MSE: %f" % nnm_mse)
#
# softImpute_mse = ((X_filled_softimpute[missing_mask] - X[missing_mask]) ** 2).mean()
# print("SoftImpute MSE: %f" % softImpute_mse)
#
# knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
# print("knnImpute MSE: %f" % knn_mse)

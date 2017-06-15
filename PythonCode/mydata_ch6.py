##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 6 - Exemplary graphs                            #
#                                                            #
##############################################################
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import math
import copy
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

from Chapter4.FrequencyAbstraction import FourierTransformation
from matplotlib.patches import Rectangle
import re
import sklearn
import numpy as np
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

##############################################################
#                                                            #
#    ROC CURVE                                               #
#                                                            #
##############################################################

# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# print (fpr)
# print(tpr)
#
# print(thresholds)

##############################################################
#                                                            #
#    Question 4                                              #
#
##############################################################
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from PythonCode.Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning

dataset_path = './intermediate_datafiles/'

dataseto = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
dataseto = dataseto.dropna(axis=0)
columnssize = len(list(dataseto))
accuracy =[]
samplesize=[]
for sample in range(1,100):
    dataset = dataseto.sample(n=sample,replace=False)
    # dataset.index = dataset.index.to_datetime()


    y = dataset[['labelWalking']]

    X = dataset.drop(dataset[['labelOnTable', 'labelSitting', 'labelWashingHands', 'labelWalking', 'labelStanding', 'labelDriving', 'labelEating', 'labelRunning']], axis=1, inplace=False)



    # print(list(y))
    # print(list(X))
    # bc = load_breast_cancer()
    # y = bc.target # ["Target"]
    # X = bc.data # [features]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # print(list(bc.feature_names))
    # print(bc.target)
    # print(list(bc.target_names))



    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X, y)

    y_predicted = dt.predict(X)
    accuracy.append( accuracy_score(y, y_predicted,normalize = True))
    samplesize.append(sample)
    # print (accuracy)
# fpr, tpr, t = roc_curve(y, results[:,0])
# plot.plot(fpr, tpr)
# plot.savefig('./Graphs/assignment2/ch6/' + 'sample' + str(int(sample)) + '.png', bbox_inches='tight')


fig = plot.figure()
plot.plot(samplesize,accuracy)
fig.suptitle("accuracy vs sample size for set with " +str(columnssize) + " columns", fontsize=20)
plot.xlabel('samplesize', fontsize=18)
plot.ylabel('accuracy', fontsize=16)

plot.show()

exit(0)

##############################################################
#                                                            #
#    part 1                                               #
#
##############################################################
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split

from PythonCode.Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning

dataset_path = './intermediate_datafiles/'

dataset = pd.read_csv(dataset_path + 'mydata_chapter5_result.csv', index_col=0)

# dataset.index = dataset.index.to_datetime()

y = dataset['labelwalking']
X = dataset[['acc_phone_x', 'acc_phone_y', 'acc_phone_z']]



# bc = load_breast_cancer()
# y = bc.target # ["Target"]
# X = bc.data # [features]
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# print(list(bc.feature_names))
# print(bc.target)
# print(list(bc.target_names))



dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X_train, y_train)
results = dt.predict_proba(X_test)
print (results)
# fpr, tpr, t = roc_curve(y_test, results[:,1])
# plot.plot(fpr, tpr)
# plot.show()

# exit(0)

fp= []
tp = []
# n= len(X_test)
for alpha in np.arange(2, 0, -0.1):
    f=0
    t=0
    for i, j in enumerate(results):
        r = j[1]
        # print(r)
        if(r > alpha):
            r = 1
        else:
            r = 0
        if (r == y_test[i]):
            t = t+1
        else:
            f = f +1
    print (alpha, t, f)
    tp.append(t)
    fp.append(f)
print (tp)
fp = np.sort(fp, axis=None)
tp = np.sort(tp, axis=None)
plot.plot(fp,tp,'r')
plot.show()
# print(y_test)

# cross_val_score(clf, bc.data, bc.target, cv=10)

# print(dt.predict())
# print(y_test)


exit(0)




##############################################################
#                                                            #
#    Gradient descent                                        #
#                                                            #
##############################################################


#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # The data to fit
# m = 20
# theta0_true = 2
# theta1_true = 0.5
# x = np.linspace(-1,1,m)
# y = theta0_true + theta1_true * x
#
# # The plot: LHS is the data, RHS will be the cost function.
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
# ax[0].scatter(x, y, marker='x', s=40, color='k')
#
# def cost_func(theta0, theta1):
#     """The cost function, J(theta0, theta1) describing the goodness of fit."""
#     theta0 = np.atleast_3d(np.asarray(theta0))
#     theta1 = np.atleast_3d(np.asarray(theta1))
#     return np.average((y-hypothesis(x, theta0, theta1))**2, axis=2)/2
#
# def hypothesis(x, theta0, theta1):
#     """Our "hypothesis function", a straight line."""
#     return theta0 + theta1*x
#
# # First construct a grid of (theta0, theta1) parameter pairs and their
# # corresponding cost function values.
# theta0_grid = np.linspace(-1,4,101)
# theta1_grid = np.linspace(-5,5,101)
# J_grid = cost_func(theta0_grid[:,np.newaxis,np.newaxis],
#                    theta1_grid[np.newaxis,:,np.newaxis])
#
# # A labeled contour plot for the RHS cost function
# X, Y = np.meshgrid(theta0_grid, theta1_grid)
# contours = ax[1].contour(X, Y, J_grid, 30)
# ax[1].clabel(contours)
# # The target parameter values indicated on the cost function contour plot
# ax[1].scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])
#
# # Take N steps with learning rate alpha down the steepest gradient,
# # starting at (theta0, theta1) = (0, 0).
# N = 5
# alpha = 0.99
# theta = [np.array((0,0))]
# J = [cost_func(*theta[0])[0]]
# for j in range(N-1):
#     last_theta = theta[-1]
#     this_theta = np.empty((2,))
#     this_theta[0] = last_theta[0] - alpha / m * np.sum(
#                                     (hypothesis(x, *last_theta) - y))
#     this_theta[1] = last_theta[1] - alpha / m * np.sum(
#                                     (hypothesis(x, *last_theta) - y) * x)
#     theta.append(this_theta)
#     J.append(cost_func(*this_theta))
#
#
# # Annotate the cost function plot with coloured points indicating the
# # parameters chosen and red arrows indicating the steps down the gradient.
# # Also plot the fit function on the LHS data plot in a matching colour.
# colors = ['b', 'g', 'm', 'c', 'orange']
# ax[0].plot(x, hypothesis(x, *theta[0]), color=colors[0], lw=2,
#            label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[0]))
# for j in range(1,N):
#     ax[1].annotate('', xy=theta[j], xytext=theta[j-1],
#                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
#                    va='center', ha='center')
#     ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=2,
#            label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
# ax[1].scatter(*zip(*theta), c=colors, s=40, lw=0)
#
# # Labels, titles and a legend.
# ax[1].set_xlabel(r'$\theta_0$')
# ax[1].set_ylabel(r'$\theta_1$')
# ax[1].set_title('Cost function')
# ax[0].set_xlabel(r'$x$')
# ax[0].set_ylabel(r'$y$')
# ax[0].set_title('Data and fit')
# axbox = ax[0].get_position()
# # Position the legend by hand so that it doesn't cover up any of the lines.
# ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
#              fontsize='small')
#
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
# exit(0)
##############################################################
#                                                            #
#    part 1                                               #
#
# ##############################################################                                                           #
np.random.seed(0)

# Figure 6.1

df = pd.DataFrame(columns=['x', 'y'])
x = np.random.normal(0, 0.5, 100)
df['x'] = x
y = 2.5 * x + 3
df['y'] = y

a = np.arange(0, 5, 0.1)
b = np.arange(0, 5, 0.1)
X, Y = np.meshgrid(a, b)

result = np.empty((0,3))
for i in b:
    for j in a:
        y_calc = x * i + j
        error = sklearn.metrics.mean_squared_error(y, y_calc)
        result = np.vstack([result, [i, j, error]])

X, Y = np.meshgrid(a, b)
e_df = pd.DataFrame(result, columns=['b', 'a', 'error'])

Z = e_df['error'].values.reshape(len(X),len(Y))

# fig = plot.figure()
# plot.hold(True)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='brg_r')
# #ax.scatter(e_df['a'], e_df['b'], e_df['error'])
# ax.set_xlabel('$\\theta_{1}$')
# ax.set_ylabel('$\\theta_{2}$')
# ax.set_zlabel('$E_{in}(h)$')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# print e_df
#
# plot.hold(False)
# plot.show()

# Figure 6.2

# plot.hold(True)
# V, U = np.gradient(Z, .2, .2)
# Q = plot.quiver(X, Y, -U, -V, pivot='mid', units='inches')
# plot.xlabel('$\\theta_{1}$')
# plot.ylabel('$\\theta_{2}$')
# plot.hold(False)
# plot.show()

for beta in np.arange(0.1, 0.9, 0.1):
    plot.hold(True)
    p = plot.contour(X, Y, Z,cmap='brg_r')
    plot.clabel(p, fontsize=9, inline=1)
    current_value = np.array([0,0])
    x_values = [0]
    y_values = [0]
    V, U = np.gradient(Z, .1, .1)
    steps = 30
    alpha =  0.1 #learning rate default 1000
    # beta =0.1
    for i in range(0, steps):
        current_value = current_value - [beta*V[current_value[0]/alpha, current_value[1]/alpha], beta*U[current_value[0]/alpha, current_value[1]/alpha]]
        x_values.append(current_value[0])
        y_values.append(current_value[1])
    plot.plot(x_values, y_values, 'k:')
    plot.gca().arrow(x_values[-1]-0.1, y_values[-1], +0.0001, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    plot.xlabel('$\\theta_{1}$')
    plot.ylabel('$\\theta_{2}$')
    plot.legend(['$gradient$ $descent$ $path$'], loc=1, fontsize='small')

    plot.hold(False)
    plot.savefig('./Graphs/assignment2/ch6/' + 'gd'+ str(int(beta*10))+'.png', bbox_inches='tight')

    plot.close()
    # plot.show()

import matplotlib.pyplot as plt
import pandas as pd
result_dataset_path = './intermediate_datafiles/'
data = pd.read_csv( result_dataset_path + 'mydata_chapter2_result.csv')

# data = pd.read_csv( result_dataset_path + 'chapter3_result_final.csv')

from scipy.stats import shapiro
import numpy as np
import pylab
import scipy.stats as stats

# normaltest(data['acc_phone_x'])
# shapiro(data['acc_phone_x'])

plt.subplot(3,3,1)
plt.title('acc_phone_x')
stats.probplot(data['acc_phone_x'], dist="norm", plot=pylab)

# pylab.show()

plt.subplot(3,3,2)
plt.title('acc_phone_y')
stats.probplot(data['acc_phone_y'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,3)
plt.title('acc_phone_z')
stats.probplot(data['acc_phone_z'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,4)
plt.title('gyr_phone_xss')
stats.probplot(data['gyr_phone_x'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,5)
stats.probplot(data['gyr_phone_y'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,6)
stats.probplot(data['gyr_phone_z'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,7)
stats.probplot(data['mag_phone_x'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,8)
stats.probplot(data['mag_phone_y'], dist="norm", plot=pylab)
# pylab.show()

plt.subplot(3,3,9)
stats.probplot(data['mag_phone_z'], dist="norm", plot=pylab)
pylab.show()


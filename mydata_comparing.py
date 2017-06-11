from __future__ import print_function
import pandas as pd
result_dataset_path = './intermediate_datafiles/'
#comparing data sets
mydata = pd.read_csv(result_dataset_path + 'mydata_chapter2_result.csv',index_col=1)
mydata.index = mydata.index.to_datetime()
del mydata['Unnamed: 0']
del mydata['press_phone_Pressure']
mydata = mydata.dropna(axis=0)

crowddata = pd.read_csv(result_dataset_path + 'chapter2_result.csv',index_col=0)
crowddata.index = crowddata.index.to_datetime()
del crowddata['hr_watch_rate']
del crowddata[ 'press_phone_pressure']
crowddata = crowddata.dropna(axis=0)

mydata_labeltable = mydata.loc[(mydata.labeltable == 1)  ]
mydata_labelsitting = mydata.loc[(mydata.labelsitting == 1)  ]
mydata_labelstanding = mydata.loc[(mydata.labelstanding == 1)  ]
mydata_labelwalking = mydata.loc[(mydata.labelwalking == 1)  ]
# mydata = mydata.query('labelsitting == 1')

# mydata = mydata.sample(n=2000)
crowddata_labeltable = crowddata.loc[(crowddata.labelOnTable == 1)  ]
crowddata_labelsitting = crowddata.loc[(crowddata.labelSitting == 1)  ]
crowddata_labelstanding = crowddata.loc[(crowddata.labelStanding == 1)  ]
crowddata_labelwalking = crowddata.loc[(crowddata.labelWalking == 1)  ]
# crowddata = crowddata.query('labelSitting == 1')

# crowddata = crowddata.sample(n=1000)

# print(list(crowddata))

# DataViz = VisualizeDataset()
#

# DataViz.plot_dataset(mydata, ['acc_', 'gyr_', 'mag_phone_', 'press_phone_', 'label'],
#                      ['like', 'like', 'like', 'like', 'like'],
#                      ['line', 'line', 'line', 'line', 'points'])
#
#
# DataViz.plot_dataset(crowddata, ['acc_', 'gyr_', 'mag_phone_', 'press_phone_', 'label'],
#                      ['like', 'like', 'like', 'like', 'like'],
#                      ['line', 'line', 'line', 'line', 'points'])

# util.print_latex_table_statistics_two_datasets(mydata,crowddata)


print(list(mydata))
print(list(crowddata))
# exit(0)
# A RankSum test will provide a P value indicating whether or not the two distributions are the same.

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


print ("MWW RankSum and anova  for mydata and crowd ")

print (" Column name  & p-val MWW RankSum & p-val_anova & ")
for col in [c for c in mydata.columns if not 'label' in c]:
    z_stat, p_val_labelOnTable = stats.ranksums(mydata_labeltable[col], crowddata_labeltable[col])
    f_val, p_val_anova_labelOnTable = stats.f_oneway(mydata_labeltable[col], crowddata_labeltable[col])

    z_stat, p_val_labelSitting = stats.ranksums(mydata_labelsitting[col], crowddata_labelsitting[col])
    f_val, p_val_anova_labelSitting = stats.f_oneway(mydata_labelsitting[col], crowddata_labelsitting[col])

    z_stat, p_val_labelStanding = stats.ranksums(mydata_labelstanding[col], crowddata_labelstanding[col])
    f_val, p_val_anova_labelStanding = stats.f_oneway(mydata_labelstanding[col], crowddata_labelstanding[col])

    z_stat, p_val_labelWalking = stats.ranksums(mydata_labelwalking[col], crowddata_labelwalking[col])
    f_val, p_val_anova_labelWalking = stats.f_oneway(mydata_labelwalking[col], crowddata_labelwalking[col])

    p_val_labelOnTable = round(p_val_labelOnTable,5)
    p_val_anova_labelOnTable = round(p_val_anova_labelOnTable,5)
    p_val_labelSitting = round(p_val_labelSitting,5)
    p_val_anova_labelSitting = round(p_val_anova_labelSitting,5)
    p_val_labelStanding = round(p_val_labelStanding,5)
    p_val_anova_labelStanding = round(p_val_anova_labelStanding,5)
    p_val_labelWalking = round(p_val_labelWalking,5)
    p_val_anova_labelWalking = round(p_val_anova_labelWalking,5)


    print (col , p_val_labelOnTable, p_val_anova_labelOnTable ,
           p_val_labelSitting ,  p_val_anova_labelSitting,
           p_val_labelStanding, p_val_anova_labelStanding,
           p_val_labelWalking , p_val_anova_labelWalking ,  sep = '&' )










# compute one-way ANOVA P value
#
#
# f_val, p_val = stats.f_oneway(mydata['acc_phone_x'], crowddata['acc_phone_x'])
#
# print ("One-way ANOVA P =", p_val)




exit(0)
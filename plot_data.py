#%% load data
import pandas as pd

train_data_file = pd.read_csv('train.csv',index_col='id')

#%% catagory features
print train_data_file.describe()
for i in xrange(51,100):
    print i
    print train_data_file.loc[:,'cat'+str(i)].describe()
    print train_data_file.loc[:,'cat'+str(i)].value_counts()
    print ' '

#%% results distribution
import numpy as np
offset = -200
raw_values = train_data_file['loss'].values
print np.mean(raw_values), np.std(raw_values)
print min(raw_values), max(raw_values)
values = np.log(raw_values[raw_values > offset] - offset)
print np.mean(values), np.std(values)
print min(values), max(values)

#%% continuous feature distribution
import matplotlib.pyplot as plt
for i in xrange(116,130):
    print ('cont', i-115)
    plt.hist(train_data_file.iloc[:,i].values, bins = 1000, color='b', normed = 1)
    plt.show()

#print sum(abs(orig_error)<2000)
#%% data size after removing outliers
print len(raw_values)
print len(values)
lowerbound = np.mean(values) - 3*np.std(values)
upperbound = np.mean(values) + 3*np.std(values)
print sum(values < lowerbound) + sum(values > upperbound)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

offset = -200
using_CV = False

def log_mae(y_predicted, y_true):
    return ('log_mae', mean_absolute_error(np.exp(np.array(y_true)),
                                           np.exp(y_predicted)))

def log_mae_feval(y_predicted, y_true):
    return ('log_mae_feval', mean_absolute_error(np.exp(np.array(y_true.get_label())),
                                           np.exp(y_predicted)))

def log_mae_score(y_predicted, y_true):
    return mean_absolute_error(np.exp(np.array(y_true)),
                                           np.exp(y_predicted))

my_loss = make_scorer(log_mae_score, greater_is_better=False)

train_data_file = pd.read_csv('train.csv',index_col='id')
test_data_file = pd.read_csv('test.csv',index_col='id')
print 'load file complete!'

values = np.log(train_data_file['loss'].values - offset)
ids_test = test_data_file.index.values

train_size = len(train_data_file)
data_file = pd.concat((train_data_file, test_data_file))
del(train_data_file, test_data_file)
print 'concat done'

# precess cont features
#v = data_file['cont14'].values
#data_file['cont14'] = (v-min(v))/(max(v)-min(v))
#data_file['cont13'] = np.log10(data_file['cont13'] + 0.1)+1
#for x in [4, 6, 7, 8, 11, 12]:
#    col_name = 'cont' + str(x)
#    data_file[col_name] = np.sqrt(data_file[col_name])

#all_features = pd.get_dummies(data_file.iloc[:,0:116]).values #iloc function eats memory!
for x in xrange(116):
    data_file['cat'+str(x+1)] = data_file['cat'+str(x+1)].astype('category').cat.codes
    #data_file.iloc[:,x] = data_file.iloc[:,x].astype('category').cat.codes
    #data_file['cat'+str(x+1)] = pd.factorize(data_file['cat'+str(x+1)])[0]
print 'encoding done'
#all_features = data_file.iloc[:,0:116].values
#np.append(all_features, data_file.iloc[:,116:130].values, axis=1)
data_file.drop('loss', axis=1, inplace=True)
all_features = data_file.values
del(data_file)
print 'features size', all_features.nbytes
print 'features shape', all_features.shape

features = all_features[:train_size]
features_test = all_features[train_size:]
del(all_features)
print 'seperation done'

#%%
reg = RandomForestRegressor(n_estimators = 200, max_features= 0.3, min_samples_leaf = 5,
                            oob_score = True, random_state = 42, n_jobs = -1)

reg.fit(features, values)
print 'Fit done'

if using_CV:
    cv_score = cross_val_score(reg, features, values, cv=5, scoring=my_loss, n_jobs=1, verbose=1)
    print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
        % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

#param_test1 = {'max_features':[0.35, 0.3, 0.25, 0.2]}
#
#gsearch1 = GridSearchCV(estimator = reg, param_grid = param_test1, verbose=2,
#                        scoring=my_loss,n_jobs=1,iid=False, cv=5)
#gsearch1.fit(features,values)
#print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#
#reg.set_params(**(gsearch1.best_params_))
#
#reg.fit(features, values)
#print 'Fit done'
values_predict = reg.predict(features)
values_test = reg.predict(features_test)
values_test = np.exp(values_test) + offset
print 'predict done'

print "Results:"
print "MAE =", log_mae(values, values_predict)

joblib.dump(reg, 'rf_model.pkl')

results = pd.DataFrame(data = values_test,
                           index = ids_test,
                           columns = ['loss'])
results.index.name = 'id'
results.to_csv('submit_file.csv', sep=',')
print 'write complete!'

print 'Training set log-scale value distribution:'
plt.hist(values, bins = 100, color='b', alpha=0.5, normed = 0)
plt.hist(values_predict, bins = 100, color='r', alpha=0.5, normed = 0)
plt.show()

#%%

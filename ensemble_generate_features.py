import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.externals import joblib

offset = -200
using_CV = True
using_grid_search = False

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
kf = KFold(n_splits=5, shuffle=True)
train_predicts = []
test_predicts = []
model_names = ['final_xgb_model.pkl',
               'xgb_model_slow.pkl',
               'xgb_model_2.pkl',
               'rf_model.pkl',
               'nn_model.pkl'
               ]
for name in model_names:
    reg = joblib.load(name)
    print reg
    #reg.set_params(n_estimators=10)
    train_predict = np.zeros(values.shape)
    test_predict = np.zeros((features_test.shape[0],))
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = values[train_idx], values[test_idx]
        reg.fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        train_predict[test_idx] = y_predict

        test_predict = test_predict + reg.predict(features_test)
        print 'one fold done'
    train_predicts.append(train_predict)
    print 'model', name, 'train predict done'
    test_predict = test_predict/5.0
    test_predicts.append(test_predict)
    print 'model', name, 'test predict done'

#%%
train_predicts.append(values)
level2_train = np.array(train_predicts)
level2_train = level2_train.T
new_train = pd.DataFrame(data = level2_train)
new_train.to_csv('level2_train.csv', sep=',')

level2_test = np.array(test_predicts)
level2_test = level2_test.T
new_test = pd.DataFrame(data = level2_test,
                           index = ids_test)
new_test.index.name = 'id'
new_test.to_csv('level2_test.csv', sep=',')
print 'write complete!'

#%%

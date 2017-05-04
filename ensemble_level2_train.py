import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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

train_data_file = pd.read_csv('level2_train.csv')
test_data_file = pd.read_csv('level2_test.csv',index_col='id')
print 'load file complete!'

features = train_data_file.iloc[:,1:-1].values
values = train_data_file.iloc[:,-1].values

features_test = test_data_file.values
ids_test = test_data_file.index.values

#%%
reg =  XGBRegressor(max_depth=3, learning_rate=0.0002, n_estimators=1000000, min_child_weight = 3,
                   objective='reg:linear', subsample=0.4, colsample_bytree=0.6,
                   silent=True, nthread=3, seed = 2016, gamma = 0.0, reg_alpha = 0.001) #params forked

if using_CV:
    xgb_param = reg.get_xgb_params()
    xgtrain = xgb.DMatrix(features, values)
    cvresult = xgb.cv(xgb_param, xgtrain,
                      num_boost_round = 10000000,
                      nfold = 5, metrics = 'rmse', feval = log_mae_feval,
                      early_stopping_rounds = 5000, verbose_eval =True, seed = 1987)
    print 'End at round', cvresult.shape[0]
    reg.set_params(n_estimators=cvresult.shape[0])

# first: max_depth, min_child_weight
# second: gamma
# third: subsample, colsample_bytree
# forth: reg_alpha
if using_grid_search:
    param_test6 = {
     'reg_alpha':[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    }
    
    gsearch1 = GridSearchCV(estimator = reg, param_grid = param_test6, verbose=2,
                            scoring=my_loss,iid=False, cv=5)
    gsearch1.fit(features,values)
    print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    
    reg.set_params(**(gsearch1.best_params_))

reg.fit(features, values, eval_metric = log_mae, verbose=True)
values_predict = reg.predict(features)
values_test = reg.predict(features_test)
values_test = np.exp(values_test) + offset

print "Results:"
print "MAE =", log_mae(values, values_predict)

joblib.dump(reg, 'level2_xgb_model.pkl')

results = pd.DataFrame(data = values_test,
                           index = ids_test,
                           columns = ['loss'])
results.index.name = 'id'
results.to_csv('level2_submit_file.csv', sep=',')
print 'write complete!'

print 'Training set log-scale value distribution:'
plt.hist(values, bins = 100, color='b', alpha=0.5, normed = 0)
plt.hist(values_predict, bins = 100, color='r', alpha=0.5, normed = 0)
plt.show()
#%%

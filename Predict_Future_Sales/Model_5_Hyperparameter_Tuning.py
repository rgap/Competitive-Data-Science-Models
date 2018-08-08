#!/usr/bin/env python3
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle

# In[2]:


# This is for hyperparameter tuning
X_train = pd.DataFrame(np.load('data/X_train.npz')['X_train'])
X_val = pd.DataFrame(np.load('data/X_val.npz')['X_val'])

y_train = np.load('data/y_train.npz')['y_train']
y_val = np.load('data/y_val.npz')['y_val']

# This is for training the final models
X = X_train.append(X_val)
y = np.append(y_train, y_val)

# In[3]:


def rmse(*args):
    """ Funcion that calculates the root mean squared error"""
    return np.sqrt(mean_squared_error(*args))
def clip20(x):
    return np.clip(x, 0, 20)


# ## Stacking approach
# 
# The final model will be an ensemble that uses the stacking approach.
# 
# So there will be 3 **base models**:
# - Light Gradient Boosting (lightgbm)
# - XGBoost
# - Random Forest
# 
# And the meta-model will be a **Linear Regression**
# 
# ## Metrics
# 
# The metric that's being used is "RMSE" because that's what has to be optimized in the final model.
# 
# ## Hyperparameter Tuning
# 
# ### A. Light Gradient Boosting
# 
# This is done so as to find the best parameters of the "lightgbm" model.

# Here I find the best learning rate by training and validating.  
# An RMSE is computed after clipping the 'y' values because that's how the results will be evaluated in the leaderboard.

# In[4]:

nthreads = 12


learning_rates = [0.02, 0.03, 0.04, 0.06, 0.08]
best_rmse = 9999999999999
for lr in learning_rates:
    lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':nthreads, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': lr, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0,
               'num_threads':nthreads,
              }

    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), int(100 * (lr / 0.03)))
    pred_lgb_val = lgb_model.predict(X_val)
    score = rmse(clip20(y_val), clip20(pred_lgb_val))

    if score < best_rmse:
        best_rmse = score
        best_lr = lr
        best_lgb = lgb_model


# In[5]:


best_lr


# #### Final training

# In[85]:


lgb_params = {
               'feature_fraction': 0.8,
               'metric': 'rmse',
               'nthread':nthreads, 
               'min_data_in_leaf': 2, 
               'bagging_fraction': 0.75, 
               'learning_rate': best_lr, 
               'objective': 'mse', 
               'bagging_seed': 100, 
               'num_leaves': 16,
               'bagging_freq':1,
               'max_depth': 16,
               'verbose':2,
               'num_threads':nthreads,
              }
best_lgb = lgb.train(lgb_params, lgb.Dataset(X, label=y),100)


# In[86]:


filename = 'models/best_lgb.sav'
pickle.dump(best_lgb, open(filename, 'wb'))


# ## Feature Importance
# 
# This is to check if features really contribute to the model.

# In[84]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# feat_importances = pd.Series(best_lgb.feature_importance(), index=X_val.columns)
# feat_importances = feat_importances.nlargest(20)
# feat_importances.plot(kind='barh')
# plt.title('Feature importance LGB')
# plt.show()


# ### B. XGBoost
# 
# Here I do a randomized grid search so that it ends faster.
# 
# Best model
# 
#     RandomizedSearchCV(cv=None, error_score='raise',
#        estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, eval_metric='rmse', gamma=0, learning_rate=0.1,
#        max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
#        n_estimators=100, n_jobs=12, nthread=None, objective='reg:linear',
#        random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#        seed=None, silent=True, subsample=1),

# In[ ]:


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# A parameter grid for XGBoost
params = {'min_child_weight':[4,5],
          'gamma':[i/10.0 for i in range(3,5)],
          'subsample':[i/10.0 for i in range(6,8)],
          'colsample_bytree':[i/10.0 for i in range(6,8)],
          'max_depth': [2,3]}

# Initialize XGB and GridSearch
xgb_model = XGBRegressor(n_jobs=nthreads, eval_metric='rmse')

train_ind=np.zeros(X.shape[0])
for i in range(0, len(X_train)):
    train_ind[i] = -1
ps = PredefinedSplit(test_fold=(train_ind))

best_xgb = RandomizedSearchCV(xgb_model, params, verbose=3, n_jobs=nthreads, cv=ps)
best_xgb.fit(X, y, verbose=True)

# best_xgb = XGBRegressor(random_state=100,
#                      n_estimators=30,
#                      max_depth=10,
#                      n_jobs=4,
#                      eval_metric='rmse')
# best_xgb.fit(X, y, verbose=True)


# In[10]:


filename = 'models/best_xgb.sav'
pickle.dump(best_xgb, open(filename, 'wb'))


# ### B. Random Forest
# 
# Trained with an increased "min_samples_leaf" because it's been overfitting.

# In[31]:


from sklearn.ensemble import RandomForestRegressor

params={'max_features':[4, 6, 8], 
        'max_depth': [2, 8],
        'min_samples_leaf' : [8, 16, 20]}

rf_model = RandomForestRegressor(n_estimators=16, n_jobs=nthreads, random_state=100, criterion='mse', verbose=2)
best_rf = RandomizedSearchCV(rf_model, params, scoring = 'neg_mean_squared_error', n_jobs=nthreads, verbose=3)
best_rf.fit(X, y)


# In[32]:


filename = 'models/best_rf.sav'
pickle.dump(best_rf, open(filename, 'wb'))


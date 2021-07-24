#!/usr/bin/env python
# coding: utf-8

# In[113]:


# pip install seaborn
#!/usr/bin/env python


# In[114]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[115]:
Path_to_data = "../data/AdSmartABdata.csv"
#Read data from csv file
db = pd.read_csv(Path_to_data, na_values=['?', None], parse_dates=['date'])

db['clicked'] = db['yes'] + db['no']

clicked_ad = db[db['clicked'] == 1]

import math
import scipy


# # **CLASSICAL A/B TESTING**

# In[126]:
exposed = db[db['clicked'] == 1][db['experiment'] == 'exposed']
control = db[db['clicked'] == 1][db['experiment'] == 'control']
exposed.sample(5)


# In[133]:


get_ipython().system('pip install statsmodels')
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


# In[134]:


db_yes = db[db['yes'] == 1]
db_yes = db_yes.drop('no', axis = 1)
db_yes = db_yes.rename(columns={"yes": "brand_awareness"})


# In[135]:


db_no = db[db['no'] == 1]
db_no = db_no.drop('yes', axis = 1)
da = {1 : 0}
db_no = db_no.replace({'no':da})
db_no = db_no.rename(columns={"no": "brand_awareness"})


# In[136]:
db_clean = pd.concat([db_yes, db_no], axis = 0)


# In[137]:
db_clean = db_clean.sample(frac=1).reset_index(drop=True)

# # **TASK 2.2**

# # Cleaning Data and Versioning
df = db_clean.copy()
df.reset_index(drop=True, inplace=True)
df.head()

# # Splitting the Data to two versions

# In[149]:


df_browser = df.drop('platform_os', axis = 1)
df_platform = df.drop('browser', axis = 1)


# In[150]:


#df_browser.to_csv('AdSmartV1.csv')
#df_platform.to_csv('AdSmartV2.csv')


# Environment

# In[151]:


pip install xgboost sklearn


# In[152]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import graphviz


# In[153]:


experiment = {'exposed':1, 'control':0}
df_browser = df_browser.replace({'experiment':experiment})

df_v1 = df_browser.drop(['device_make'], axis = 1)

df_v1.head()


# In[154]:


encoder = OneHotEncoder()
encode = encoder.fit_transform(df_v1[["browser"]])

encoded = encode.toarray()

new_encoded = np.sum(encoded, axis = 1, dtype = int)

df_v1.head()


# In[155]:


train_browser, test_browser, validate_browser  =               np.split(df_browser[df_browser.columns[1:]].sample(frac=1, random_state=42), 
                       [int(.7*len(df_browser)), int(.9*len(df_browser))])


# In[156]:


browser_X_col = df_browser.columns[1:-1]

browser_train =  train_browser[browser_X_col]
browser_train_labels = train_browser['brand_awareness']

browser_test =  test_browser[browser_X_col]
browser_test_labels = test_browser['brand_awareness']

browser_val =  validate_browser[browser_X_col]
browser_val_labels = validate_browser['brand_awareness']


# In[157]:


numerical_browser = ['date','hour', 'experiment']
cat_browser = ['device_make', 'browser']


# In[158]:


class ExtractDay(BaseEstimator, TransformerMixin):
    def __init__(self): 
        pass
    def fit(self, X, y=None):
        return self               
    
    def transform(self, X, y=None):
        X= X.copy()
        X['date'] = X['date'].apply(lambda x:x.weekday())
        return X


# In[159]:


num_pipeline = Pipeline([
    ('day_extractor', ExtractDay()),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numerical_browser),
    ('cat', OneHotEncoder(), numerical_browser)
])


# In[160]:


browser_train_processed = full_pipeline.fit_transform(browser_train)
browser_test_processed = full_pipeline.fit_transform(browser_test)
browser_val_processed = full_pipeline.fit_transform(browser_val)


# In[161]:


params ={'objective':'binary:logistic', 'learning_rate':0.1, 'max_depth':5, 'random_state':42, 'use_label_encoder':False, 'nfold':5}


# In[162]:


clf_xgb = xgb.XGBClassifier(base_score=0.5, max_depth=5, learning_rate=0.1, random_state=42, objective='binary:logistic', use_label_encoder=False, nfold=5)


# In[163]:


clf_xgb.fit(browser_test_processed, browser_test_labels)


# In[164]:


y_predicted = clf_xgb.predict(browser_test_processed)


# In[165]:


rms = mean_squared_error(browser_test_labels, y_predicted, squared=False)
rms


# In[166]:


data_dmatrix = xgb.DMatrix(data=browser_val_processed, label=browser_val_labels)


# In[167]:


cv_result = xgb.cv(params=params,dtrain=data_dmatrix, nfold=5, metrics='rmse', stratified=True, as_pandas=True, seed=42)
cv_result


# In[168]:


xgb.plot_importance(clf_xgb)
plt.rcParams['figure.figsize'] = [20, 20]


# # PLATFORM

# In[169]:


experiment = {'exposed':1, 'control':0}
df_platform = df_platform.replace({'experiment':experiment})


# In[170]:


train_platform, test_platform, validate_platform  =               np.split(df_platform[df_platform.columns[1:]].sample(frac=1, random_state=42), 
                       [int(.7*len(df_platform)), int(.9*len(df_platform))])


# In[171]:


platform_X_col = df_platform.columns[1:-1]

platform_train =  train_platform[platform_X_col]
platform_train_labels = train_platform['brand_awareness']

platform_test =  test_platform[platform_X_col]
platform_test_labels = test_platform['brand_awareness']

platform_val =  validate_platform[platform_X_col]
platform_val_labels = validate_platform['brand_awareness']


# In[172]:


numerical_platform = ['date','hour', 'experiment']
cat_platform = ['device_make', 'browser']


# In[173]:


class ExtractDay(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self               
    
    def transform(self, X, y=None):
        X= X.copy()
        X['date'] = X['date'].apply(lambda x:x.weekday())
        return X


# In[174]:


numerical_platform= ['date','hour','platform_os', 'experiment']
cat_platform = ['device_make']


# In[175]:


num_pipeline = Pipeline([
    ('day_extractor', ExtractDay()),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numerical_platform),
    ('cat', OneHotEncoder(), numerical_platform)
])


# In[176]:


platform_train_processed = full_pipeline.fit_transform(platform_train)
platform_test_processed = full_pipeline.fit_transform(platform_test)
platform_val_processed = full_pipeline.fit_transform(platform_val)


# In[177]:


params ={'objective':'binary:logistic', 'learning_rate':0.1, 'max_depth':5, 'random_state':42, 'use_label_encoder':False, 'nfold':5}


# In[178]:


clf_xgb = xgb.XGBClassifier(base_score=0.5, max_depth=5, learning_rate=0.1, random_state=42, objective='binary:logistic', use_label_encoder=False, nfold=5)


# In[179]:


clf_xgb.fit(platform_test_processed, platform_test_labels)


# In[180]:


y_predicted = clf_xgb.predict(platform_test_processed)


# In[181]:


rms = mean_squared_error(platform_test_labels, y_predicted, squared=False)
rms


# In[182]:


data_dmatrix = xgb.DMatrix(data=platform_val_processed, label=platform_val_labels)


# In[183]:


cv_result = xgb.cv(params=params,dtrain=data_dmatrix, nfold=5, metrics='rmse', stratified=True, as_pandas=True, seed=42)
cv_result


# In[184]:


xgb.plot_importance(clf_xgb)
plt.rcParams['figure.figsize'] = [20, 20]





# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:23:14 2019

@author: DEBAL
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# In[]
df_train_x = pd.read_csv("Dataset/train.csv")
df_train_y = pd.read_csv("Dataset/train_label.csv", header=None)
df_test_x = pd.read_csv("Dataset/test.csv")
df_test_y = pd.read_csv("Dataset/test_label.csv", header=None)

df_train_x_new = df_train_x.copy()
df_test_x_new = df_test_x.copy()

heat_map = sb.heatmap(df_train_x[['temp','atemp','humidity','windspeed']],cmap="YlGnBu")
plt.show()

plt.figure()
sb.boxplot(x=df_train_x['season'], y=df_train_x['temp'], data=df_train_x)
plt.show()

plt.figure()
plt.scatter(df_train_x.temp, df_train_y.values)
plt.show()

plt.figure()
plt.scatter(df_train_x.atemp, df_train_y.values)
plt.show()

plt.figure()
plt.scatter(df_train_x.humidity, df_train_y.values)
plt.show()

plt.figure()
plt.scatter(df_train_x.windspeed, df_train_y.values)
plt.show()


# In[]:

outliers = df_train_x.loc[(df_train_x['season'] == 'Fall') & (df_train_x['temp'] < 20)]
df_train_x_temp = df_train_x.drop(outliers.index)

# find missing values
#print(np.sum(df_train_x.isna()))
#print(np.sum(df_train_x.isnull()))
# find entries in a given range
#print(np.sum(df_train_x.isin([0,1])))

#plt.close('all')

# In[]:

train_seasonCat = LabelBinarizer().fit_transform(df_train_x["season"])
train_weatherCat = LabelBinarizer().fit_transform(df_train_x["weather"])

test_seasonCat = LabelBinarizer().fit_transform(df_test_x["season"])
test_weatherCat = LabelBinarizer().fit_transform(df_test_x["weather"])

df_train_x_new['datetime'] = pd.to_datetime(df_train_x.datetime)
df_test_x_new['datetime'] = pd.to_datetime(df_test_x.datetime)

# In[]:
# extract datetime elements
year = []
month = []
day = []
hour = []
minute = []
sec = []
for dt in df_train_x_new.datetime:
  year.append(dt.year)
  month.append(dt.month)
  day.append(dt.day)
  hour.append(dt.hour)
  minute.append(dt.minute)
  sec.append(dt.second)

df_train_x_new['year'] = year
df_train_x_new['month'] = month
df_train_x_new['day'] = day
df_train_x_new['hour'] = hour
df_train_x_new['minute'] = minute
df_train_x_new['sec'] = sec

df_train_x_new.drop('datetime', axis=1, inplace=True)

# In[]:
# extract datetime elements
year = []
month = []
day = []
hour = []
minute = []
sec = []
for dt in df_test_x_new.datetime:
  year.append(dt.year)
  month.append(dt.month)
  day.append(dt.day)
  hour.append(dt.hour)
  minute.append(dt.minute)
  sec.append(dt.second)

df_test_x_new['year'] = year
df_test_x_new['month'] = month
df_test_x_new['day'] = day
df_test_x_new['hour'] = hour
df_test_x_new['minute'] = minute
df_test_x_new['sec'] = sec

df_test_x_new.drop('datetime', axis=1, inplace=True)


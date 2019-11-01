# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:23:14 2019

@author: DEBAL
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
# drop outliers
outliers = df_train_x.loc[(df_train_x['season'] == 'Fall') & (df_train_x['temp'] < 20)]
df_train_x_temp = df_train_x.drop(outliers.index)

# find missing values
#print(np.sum(df_train_x.isna()))
#print(np.sum(df_train_x.isnull()))
# find entries in a given range
#print(np.sum(df_train_x.isin([0,1])))

# In[]:
LB_season = LabelBinarizer()
LB_weather = LabelBinarizer()
train_seasonCat = LB_season.fit_transform(df_train_x["season"])
train_weatherCat = LB_weather.fit_transform(df_train_x["weather"])

test_seasonCat = LB_season.transform(df_test_x["season"])
test_weatherCat = LB_weather.transform(df_test_x["weather"])

df_train_x_new['datetime'] = pd.to_datetime(df_train_x.datetime)
df_test_x_new['datetime'] = pd.to_datetime(df_test_x.datetime)

# In[]:
# extract datetime elements
year_train = []
month_train = []
day_train = []
hour_train = []
minute_train = []
sec_train = []
for dt in df_train_x_new.datetime:
  year_train.append(dt.year)
  month_train.append(dt.month)
  day_train.append(dt.day)
  hour_train.append(dt.hour)
  minute_train.append(dt.minute)
  sec_train.append(dt.second)

df_train_x_new['year'] = year_train
df_train_x_new['month'] = month_train
df_train_x_new['day'] = day_train
df_train_x_new['hour'] = hour_train
df_train_x_new['minute'] = minute_train
df_train_x_new['sec'] = sec_train

df_train_x_new.drop('datetime', axis=1, inplace=True)

# In[]:
# extract datetime elements
year_test = []
month_test = []
day_test = []
hour_test = []
minute_test = []
sec_test = []
for dt in df_test_x_new.datetime:
  year_test.append(dt.year)
  month_test.append(dt.month)
  day_test.append(dt.day)
  hour_test.append(dt.hour)
  minute_test.append(dt.minute)
  sec_test.append(dt.second)

df_test_x_new['year'] = year_test
df_test_x_new['month'] = month_test
df_test_x_new['day'] = day_test
df_test_x_new['hour'] = hour_test
df_test_x_new['minute'] = minute_test
df_test_x_new['sec'] = sec_test

df_test_x_new.drop('datetime', axis=1, inplace=True)

# In[]
# Prepare training data
train_len = len(df_train_x_new)

Train_X = np.hstack([np.reshape(year_train,(train_len,1)), 
                     np.reshape(month_train,(train_len,1)),
                     np.reshape(day_train,(train_len,1)),
                     np.reshape(hour_train,(train_len,1)),
                     np.reshape(minute_train,(train_len,1)),
                     np.reshape(sec_train,(train_len,1)),
                     train_seasonCat,
                     np.reshape(list(df_train_x_new['holiday']),(train_len,1)),
                     np.reshape(list(df_train_x_new['workingday']),(train_len,1)),
                     train_weatherCat,
                     np.reshape(list(df_train_x_new['temp']),(train_len,1)),
                     np.reshape(list(df_train_x_new['atemp']),(train_len,1)),
                     np.reshape(list(df_train_x_new['humidity']),(train_len,1)),
                     np.reshape(list(df_train_x_new['windspeed']),(train_len,1)),
                     ])

Train_Y = np.array(df_train_y)

# In[]
# Prepare test data
test_len = len(df_test_x_new)

Test_X = np.hstack([np.reshape(year_test,(test_len,1)), 
                     np.reshape(month_test,(test_len,1)),
                     np.reshape(day_test,(test_len,1)),
                     np.reshape(hour_test,(test_len,1)),
                     np.reshape(minute_test,(test_len,1)),
                     np.reshape(sec_test,(test_len,1)),
                     test_seasonCat,
                     np.reshape(list(df_test_x_new['holiday']),(test_len,1)),
                     np.reshape(list(df_test_x_new['workingday']),(test_len,1)),
                     test_weatherCat,
                     np.reshape(list(df_test_x_new['temp']),(test_len,1)),
                     np.reshape(list(df_test_x_new['atemp']),(test_len,1)),
                     np.reshape(list(df_test_x_new['humidity']),(test_len,1)),
                     np.reshape(list(df_test_x_new['windspeed']),(test_len,1)),
                     ])

Test_Y = np.array(df_test_y)

# In[]
# Train models

#trainx, testx, trainy, testy = train_test_split(Train_X, Train_Y, test_size=0.25, random_state=42)

#model = LinearRegression(normalize=True)
#model = LinearSVR(verbose=1, max_iter=2000)
#model = AdaBoostRegressor()
model = RandomForestRegressor()

H = model.fit(Train_X, Train_Y)

pred_y = model.predict(Test_X)
pred_y = np.reshape(pred_y, (len(pred_y),1))

# calculate root mean square error
rmse = np.sqrt(metrics.mean_squared_error(Test_Y, pred_y))

print(rmse)

# In[]
# close all figures
#plt.close('all')

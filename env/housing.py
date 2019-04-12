# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"pycharm": {}}
import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# + {"pycharm": {}}
fetch_housing_data()

# + {"pycharm": {"is_executing": false}}
HOUSING_PATH = "datasets/housing"
import pandas as pd
import os
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
load_housing_data().head()

# + {"pycharm": {"is_executing": false}}
housing = load_housing_data()
housing.info()

# + {"pycharm": {}}
housing["ocean_proximity"].value_counts()

# + {"pycharm": {"is_executing": false}}
housing.describe()

# + {"pycharm": {"is_executing": false}}
# %matplotlib inline   
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

# + {"pycharm": {"is_executing": false}}
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    train_indices = shuffled_indices[:test_set_size]
    test_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# + {"pycharm": {"is_executing": false}}
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

## 使用scikit-learn工具进行数据集切割

# + {"pycharm": {"metadata": false, "name": "#%%\n", "is_executing": false}}
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, train_size=0.3, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


# + {"pycharm": {"metadata": false, "name": "#%%\n", "is_executing": false}}
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"]< 5,5.0,inplace=True)
'''
inplace=False
3.0     0.350581
2.0     0.318847
4.0     0.176308
5.0     0.068944
1.0     0.039826
6.0     0.025775
7.0     0.009157
8.0     0.005087
9.0     0.002422

inplace=True
'''
print(housing["income_cat"].tail())

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
housing["income_cat"].value_counts() / len(housing)
# -
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# alpha可视化参数： 根据密度进行透视处理
housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)

# population 人口数， 颜色区分价格
import matplotlib.pyplot as plt
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/1000, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
)
plt.legend()


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)

# +
from pandas.tools.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# -



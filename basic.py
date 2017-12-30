# https://www.kaggle.com/ravaliraj/titanic-data-visualization-and-ml

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import logging
import os
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

input_folder = './input/'
for item in os.listdir(input_folder):
    logger.debug('%s contains file or subfolder %s' % (input_folder, item))

sns.set_style('whitegrid')

train_file = input_folder + 'train.csv'
logger.debug('data load from %s begin.' % train_file)
titanic_df = pd.read_csv(train_file)
logger.debug('data load complete.')
logger.debug(titanic_df.head())

# todo find a way to do this that allows us to log it rather than have it print directly to stdout
# titanic_df.info()
logger.debug(titanic_df.describe())

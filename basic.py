# https://www.kaggle.com/ravaliraj/titanic-data-visualization-and-ml

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# from sklearn.cross_validation import train_test_split

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

# plot gender/sex as a factor plot
sns.factorplot('Sex', data=titanic_df, kind='count')
sex_factorplot_file = './titanic_sex_factorplot.png'
logger.debug('saving sex factor plot as %s' % sex_factorplot_file)
plt.savefig(sex_factorplot_file)

# plot class as a factor plot
sns.factorplot('Pclass', data=titanic_df, kind='count')
class_factorplot_file = './titanic_class_factorplot.png'
logger.debug('saving class factor plot as %s' % class_factorplot_file)
plt.savefig(class_factorplot_file)

# plot class and sex together
sns.factorplot('Pclass', data=titanic_df, hue='Sex', kind='count')
class_sex_factorplot_file = './titanic_class_sex_factorplot.png'
logger.debug('saving sex/class factor plot as %s' % class_sex_factorplot_file)
plt.savefig(class_sex_factorplot_file)

# age histogram
figure = plt.figure()
titanic_df['Age'].hist(bins=70)
age_histogram_file = './age_histogram.png'
logger.debug('saving age histogram as %s' % age_histogram_file)
plt.savefig(age_histogram_file)
plt.close(figure)
del figure

# add a facet grid that shows the age by sex
figure = plt.figure()
age_sex_figure = sns.FacetGrid(titanic_df, hue='Sex', aspect=5)
age_sex_figure.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
age_sex_figure.set(xlim=(0, oldest))
age_sex_figure.add_legend()
age_sex_facet_file = './age_sex_facets.png'
logger.debug('saving age/sex facet KDEplot as %s' % age_sex_facet_file)
plt.savefig(age_sex_facet_file)
plt.close(figure)
del figure

# get the mean age
mean_age = titanic_df['Age'].mean()
logger.debug('mean passenger age: %.1f' % mean_age)

# embarked factor plot
figure = plt.figure()
sns.factorplot('Embarked', data=titanic_df, kind='count')
embarked_factorplot_file = './embarked_factorplot.png'
logger.debug('saving embarked factor plot to %s' % embarked_factorplot_file)
plt.savefig(embarked_factorplot_file)
plt.close(figure)
del figure

# embarked factor plot broken out by class
figure = plt.figure()
sns.factorplot('Embarked', data=titanic_df, hue='Pclass', kind='count')
embarked_by_class_factorplot_file = './embarked_by_class_factorplot.png'
logger.debug('saving embarked/class factor plot to %s' % embarked_by_class_factorplot_file)
plt.savefig(embarked_by_class_factorplot_file)
plt.close(figure)
del figure

# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp
# I know what I'm doing here, so turn  off this warning temporarily
pd.options.mode.chained_assignment = None
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Without Family'
pd.options.mode.chained_assignment = 'warn'
figure = plt.figure()
sns.factorplot('Alone', kind='count', data=titanic_df)
alone_factorplot_file = './alone_factorplot.png'
logger.debug('saving alone/family factor plot to %s' % alone_factorplot_file)
plt.savefig(alone_factorplot_file)
plt.close(figure)
del figure

# survived factor plot
figure = plt.figure()
sns.factorplot('Survived', data=titanic_df, kind='count')
survived_factorplot_file = './survived_factorplot.png'
logger.debug('saving survived factor plot to %s' % survived_factorplot_file)
plt.savefig(survived_factorplot_file)
plt.close(figure)
del figure

# survived x class factor plot
figure = plt.figure()
sns.factorplot('Survived', data=titanic_df, kind='count', hue='Pclass')
survived_class_factorplot_file = './survived_class_factorplot.png'
logger.debug('saving survived/class factor plot to %s' % survived_class_factorplot_file)
plt.close(figure)
plt.savefig(survived_class_factorplot_file)
del figure

# survived x sex factor plot
figure = plt.figure()
sns.factorplot('Pclass', 'Survived', data=titanic_df, hue='Sex')
survived_sex_factorplot_file = './survived_sex_factorplot.png'
logger.debug('saving survived/sex factor plot to %s' % survived_sex_factorplot_file)
plt.savefig(survived_sex_factorplot_file)
plt.close(figure)
del figure

# age x survived lmplot
figure = plt.figure()
sns.lmplot('Age', 'Survived', data=titanic_df)
age_survived_lmplot_file = './age_survived_lmplot.png'
logger.debug('saving age/survived lm plot to %s' % age_survived_lmplot_file)
plt.savefig(age_survived_lmplot_file)
plt.close(figure)
del figure

# age x survived x class lmplot
figure = plt.figure()
sns.lmplot('Age', 'Survived', data=titanic_df, hue='Pclass')
age_survived_class_lmplot_file = './age_survived_class_lmplot.png'
logger.debug('saving age/survived/class lm plot to %s' % age_survived_class_lmplot_file)
plt.savefig(age_survived_class_lmplot_file)
plt.close(figure)
del figure

# drop the cabin
titanic_df.drop('Cabin', axis=1, inplace=True)

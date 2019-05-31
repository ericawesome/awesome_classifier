# 0. Import libraries and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pylab import savefig
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix
''' & to surpresse warning messages '''
import warnings
warnings.filterwarnings('ignore')

titanic_data = pd.read_csv('train.csv', index_col=0)


# 1) Feature engineering
def feature_engineer(dataframe):
    ''' (A) As a value for the NaN in Age as input the mean age of the pclass passenger '''
    mean_age_pclass1 = dataframe[dataframe['Pclass'] == 1]['Age'].mean()
    mean_age_pclass2 = dataframe[dataframe['Pclass'] == 2]['Age'].mean()
    mean_age_pclass3 = dataframe[dataframe['Pclass'] == 3]['Age'].mean()

    age_list = []
    for i, row in dataframe.iterrows():
        if row.isnull()['Age']:
            if row['Pclass'] == 1:
                age_list.append(mean_age_pclass1)
            elif row['Pclass'] == 2:
                age_list.append(mean_age_pclass2)
            else:
                age_list.append(mean_age_pclass3)
        else:
            age_list.append(row['Age'])

    dataframe['Age'] = age_list

    ''' (B) Binning ages (0-12|13-25|26-39|40-53|54-67|68-80) and label with numbers '''
    dataframe['Age'] = (pd.cut(dataframe['Age'], 6, labels=range(6))).astype(int)

    ''' (C) Converting strings into numerical data: Sex, Embarked '''
    dataframe['Sex'] = pd.get_dummies(dataframe['Sex'], drop_first=True)
    d = {'S': 0, 'C': 1, 'Q': 2}
    dataframe['Embarked'] = (dataframe['Embarked'].map(d).fillna(method='ffill')).astype(int)

    ''' (D) Extract dataframe with columns in interest'''
    dataframe = dataframe.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'])

    return dataframe

featured_data = feature_engineer(titanic_data)


# 2. Data visualization
''' Compute correlation matrix of featured dataframe with seaborn'''
corr_features = featured_data.corr()

mask = np.zeros_like(corr_features, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(8, 6))
sns.heatmap(corr_features, mask=mask, vmax=0.5, vmin=0, center=0, cmap='Blues',
            annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .6})

''' Plot probabilities to survive due to Pclass, Sex, Age, Embarked, Family Members '''
(featured_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)).plot.bar()
(featured_data[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)).plot.bar()
(featured_data[['Age', 'Survived']].groupby(['Age']).mean().sort_values(by='Survived', ascending=False)).plot.bar()
(featured_data[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)).plot.bar()
None


# 3. Train logistic regression model (RandomForestClassifier)
''' Train / Test split '''
X = featured_data.drop(columns=['Survived'])
y = featured_data['Survived']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape

rdm_classifier = RandomForestClassifier()
rdm_classifier.fit(Xtrain, ytrain)
print(f'\nScore with training data using RandomForest with default estimators: {rdm_classifier.score(Xtrain, ytrain)}')
print(f'Score with test data using RandomForest with default estimators: {rdm_classifier.score(Xtest, ytest)} \n -----------------------------------')


# 4. Hyperparameter optimization (GridSearchCV)
my_param_grid = {'n_estimators': range(1, 11), 'max_depth': range(1, 11)}
grid = GridSearchCV(rdm_classifier, param_grid=my_param_grid)
grid.fit(Xtrain, ytrain)
rdm_optimizer = grid.best_estimator_
print(f'\nScore with training data using RandomForest with best estimators: {grid.best_score_}')
print(f'Best parameters: {grid.best_params_} \n -----------------------------------')


# 5. Applying new hyperparameter on model (RandomForestClassifier)
'''Using the parameters from the GridSeachCV for an optimized model'''
rdm_classifier_opt = rdm_optimizer
rdm_classifier_opt.fit(Xtrain, ytrain)
print(f'\n\nScore with training data using optimized RandomForest model: {rdm_classifier_opt.score(Xtrain, ytrain)}')
print(f'Score with test data using optimized RandomForest model: {rdm_classifier_opt.score(Xtest, ytest)} \n -----------------------------------')

'''Print statsmodel summary'''
X2 = sm.add_constant(Xtrain)
logit_model = sm.Logit(ytrain, X2)
result = logit_model.fit()
print(result.summary2())


# 6. Print confusion matrix as evaluation of the model
ypred = rdm_classifier_opt.predict(Xtrain)

print(f'\nPrediction accuracy on "Passengers that were predicted to be alive but unfortunately died": {precision_score(ypred, ytrain)}')
print(f'The goal is to achieve a high recall score: {recall_score(ypred, ytrain)} because you want to minimize the possibility \nthat passengers were not in the focus of rescue activities because they were wrongly classified by the model! \n\n')
print(confusion_matrix(ypred, ytrain))

plt.figure(figsize=(8, 6))
svm = sns.heatmap(confusion_matrix(ypred, ytrain), annot=True, fmt='4g', linewidths=.5, xticklabels=['Survived(true)', 'Dead(true)'], yticklabels=['Survived(predicted)', 'Dead(predicted)'])
plt.show()

figure = svm.get_figure()
figure.savefig('svm_heatmap.png', dpi=400)

print(f'\n\nThank you for running the code. A confusion-matrix was printed and saved as svm_heatmap!')
print(f'Try the model on your test data and find out how you score on: https://www.kaggle.com/c/titanic/submissions \n -----------------------------------')

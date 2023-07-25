import pandas as pd
from numpy import mean
from numpy import std
from sklearn.impute import KNNImputer
from numpy import isnan
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from numpy import nan

#loading dataset
dataset = pd.read_csv('health care employment.csv', header=None, na_values='?')
#check if values missing were replaced
print(dataset.head())

#split into input and output elements
data = dataset.values
ix=[i for i in range(data.shape[1]) if i!=23]
X,y = data[:,ix], data[:, 23]
#summarize the number of rows with missing values for each column
#iterate through the columns, counting for each column how many null values there are

#ignore future warning bc of syntax of line 24
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

for i in range (dataset.shape[1]):
    #count the number of rows that contain missing values
    n_miss = dataset[[i]].isnull().sum()
    perc = n_miss / dataset.shape[0] * 100
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
#using knn imputer
imputer = KNNImputer(n_neighbors = 5, weights = 'uniform', metric = 'nan_euclidean')
#fit imputer on dataset
imputer.fit(X)
#transform the dataset
Xtrans = imputer.transform(X)
#total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))

#modeling pipeline
model = RandomForestClassifier()
imputer = KNNImputer()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
#model evaluation
cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state = 1)
#final evaluation
scores = cross_val_score(pipeline ,X, y, scoring = 'accuracy', cv=cv, n_jobs = -1, error_score = 'raise' )
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#evaluating various k values/strategies on the dataset
results = list()
strategies = [str(i) for i in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
for s in strategies:
    #modeling pipeline
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())])
    #evaluate model
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    #store results
    results.append(scores)
    print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
    # plot model performance for comparison
pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()

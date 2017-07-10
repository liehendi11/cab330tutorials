# coding: utf-8
import pandas as pd
import numpy as np

# read the pva97nk dataset
df = pd.read_csv('pva97nk.csv')

# drop ID and the unused target variable
df.drop(['ID', 'TargetD'], axis=1, inplace=True)

# impute missing values in DemAge with its mean
df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

# change DemCluster from interval/integer to nominal/str
df['DemCluster'] = df['DemCluster'].astype(str)

# change DemHomeOwner into binary 0/1 variable
dem_home_owner_map = {'U':0, 'H': 1}
df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)

# denote miss values in DemMidIncome
temp = df['DemMedIncome']
temp[temp < 1] = 0
df['DemMedIncome'] = temp

df['DemMidIncome'].replace(0, np.nan, inplace=True)

# impute med income using average strategy
df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

# impute gift avg card 36 using average strategy
df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)
import pandas as pd
import numpy as np

# read the pva97nk dataset
df = pd.read_csv('pva97nk.csv')

# drop ID and the unused target variable
df.drop(['ID', 'TargetD'], axis=1, inplace=True)

# impute missing values in DemAge with its mean
df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

# change DemCluster from interval/integer to nominal/str
df['DemCluster'] = df['DemCluster'].astype(str)

# change DemHomeOwner into binary 0/1 variable
dem_home_owner_map = {'U':0, 'H': 1}
df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)

# denote miss values in DemMidIncome
temp = df['DemMedIncome']
temp[temp < 1] = 0
df['DemMedIncome'] = temp

df['DemMedIncome'].replace(0, np.nan, inplace=True)

# impute med income using average strategy
df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

# impute gift avg card 36 using average strategy
df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)
from sklearn.model_selection import train_test_split
y = df['TargetB']
X = df.drop(['TargetB'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.5, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.train(X_train, y_train)
model.fit(X_train, y_train)
df
df2 = pd.get_dummies(df)
df2
y = df2['TargetB']
X = df2.drop(['TargetB'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(3, 10), 'min_samples_leaf': range(20, 200, 20)}
cv = GridSearchCV(param_grid=params, model=DecisionTreeClassifier(), cv=5)
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
cv.best_params_
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 5), 'min_samples_leaf': range(10, 30, 10)}
cv.fit(X_train, y_train)
get_ipython().magic('clear ')
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
params = {'max_leaf_nodes': range(1,15)}
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
params = {'max_leaf_nodes': range(2,15)}
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
model = cv.best_estimator_
model.feature_importances_
model
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 5), 'min_samples_leaf': range(10, 30, 10)}
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
cv.best_estimator_.feature_importances_
imp = cv.best_estimator_.feature_importances_
imp[0]
names = X.columns
names[0]
indices = np.argsort(importances)indices = np.flip(indices, axis=0)
indices = np.argsort(importances)indices = np.flip(indices, axis=0)
indices = np.argsort(importances)indices = np.flip(indices, axis=0)
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)
indices = np.argsort(imp)
indices = np.flip(indices, axis=0)
indices
for i in indices:
    print names[i],':',imp[i]
for i in indices:
    print(names[i],':',imp[i])
    

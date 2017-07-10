import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from dm_tools import data_prep
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

# preprocessing step

df = data_prep()

# train test split
y = df['TargetB']
X = df.drop(['TargetB'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.5, random_state=42)

# simple decision tree training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# grid search CV
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(3, 10), 'min_samples_leaf': range(20, 200, 20)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print(cv.best_params_)

# grid search CV #2
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 5), 'min_samples_leaf': range(10, 30, 10)}
cv.fit(X_train, y_train)
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# grid search on max leaf nodes only
params = {'max_leaf_nodes': range(2, 15)}
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# feature importances, retrain the CV first
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 5), 'min_samples_leaf': range(10, 30, 10)}
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=5)
cv.fit(X_train, y_train)

importances = cv.best_estimator_.feature_importances_
names = X.columns

indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

for i in indices:
    print(names[i], ':', importances[i])


# visualize
model = cv.best_estimator_
dotfile = StringIO()
export_graphviz(model, out_file=dotfile, feature_names=names)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_png("week3_dt_viz.png")

# coding: utf-8
import pandas as pd
import numpy as np

def data_prep():
    # read the pva97nk dataset
    df = pd.read_csv('pva97nk.csv')

    # drop ID and the unused target variable
    df.drop(['ID', 'TargetD'], axis=1, inplace=True)

    # impute missing values in DemAge with its mean
    df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

    # change DemCluster from interval/integer to nominal/str
    df['DemCluster'] = df['DemCluster'].astype(str)

    # change DemHomeOwner into binary 0/1 variable
    dem_home_owner_map = {'U': 0, 'H': 1}
    df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)

    # denote miss values in DemMidIncome
    mask = df['DemMedIncome'] < 1
    df.loc[mask, 'DemMedIncome'] = np.nan

    # df['DemMedIncome'].replace(0, np.nan, inplace=True)

    # impute med income using average strategy
    df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

    # impute gift avg card 36 using average strategy
    df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)

    # one hot encoding
    df = pd.get_dummies(df)

    return df
data_prep()
df = data_prep()
df['DemMedIncome']
y = df['TargetB']
X = df.drop(['TargetB'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.5, random_state=42)
from sklearn.model_selection import train_test_split
y = df['TargetB']
X = df.drop(['TargetB'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.5, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.feature_selection import RFE
sel = RFE(LogisticRegression())
sel.fit(X_train, y_train)
y_pred = sel.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
y = df['TargetB']
X = df.drop(['TargetB'], axis=1)
X_mat = normalize(X.as_matrix())
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.5, random_state=42)
X_train
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
for i in range(2, len(names)+1):
    select = SelectKBest(score_func=f_classif, k=i)
    X_transf = select.fit_transform(X_mat, y)
    X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size=0.5, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(i, accuracy_score(y_test, y_pred))
    
names = df.columns
for i in range(2, len(names)+1):
    select = SelectKBest(score_func=f_classif, k=i)
    X_transf = select.fit_transform(X_mat, y)
    X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size=0.5, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(i, accuracy_score(y_test, y_pred))
    
for i in range(2, len(names)):
    select = SelectKBest(score_func=chi2, k=i)
    X_transf = select.fit_transform(X_mat, y)
    X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size=0.5, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(i, accuracy_score(y_test, y_pred))
    

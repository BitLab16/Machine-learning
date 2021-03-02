import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score 
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def heatmap(data):
    data.corr()  # correlazione tra variabili in tabella n * n
    hm = sns.heatmap(data.corr(), cbar=True, square=False, yticklabels=data.columns, xticklabels=data.columns)
    plt.show()
    data.head()


def scaledata(data):
    transformers = [
        ['one_hot', OneHotEncoder(), ['season', 'yr', 'mnth', 'weekday', 'weathersit', 'hr']],
        ['scaler', StandardScaler(), ['temp', 'atemp', 'hum', 'windspeed', 'holiday', 'workingday']]
    ]
    ct = ColumnTransformer(transformers, remainder="passthrough")
    X = ct.fit_transform(data)
    X = data.drop(["cnt"], axis=1)  # features
    y = data["cnt"]  # labels
    return X,y

def rf(X_train, X_test, y_train, y_test):
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rf_model.fit(X_train, y_train)
    p_train = rf_model.predict(X_train)
    p_test = rf_model.predict(X_test)
    mae_train = mean_absolute_error(y_train, p_train)
    mae_test = mean_absolute_error(y_test, p_test)
    print("R2 score: " + str(r2_score(y_test, p_test)))
    res = pd.DataFrame({'Actual': y_test, 'Predicted': p_test})
    pred = pd.DataFrame(data=p_test)
    pred.to_csv('pred.csv')


def dt(X_train, X_test, y_train, y_test):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    p_train = dt_model.predict(X_train)
    p_test = dt_model.predict(X_test)
    mae_train = mean_absolute_error(y_train, p_train)
    mae_test = mean_absolute_error(y_test, p_test)
    print("R2 score: " + str(r2_score(y_test, p_test)))
    res = pd.DataFrame({'Actual': y_test, 'Predicted': p_test})
    pred = pd.DataFrame(data=p_test)
    pred.to_csv('pred.csv')


def ada(X_train, X_test, y_train, y_test):
    ada_model = AdaBoostRegressor()
    ada_model.fit(X_train, y_train)
    p_train = ada_model.predict(X_train)
    p_test = ada_model.predict(X_test)
    mae_train = mean_absolute_error(y_train, p_train)
    mae_test = mean_absolute_error(y_test, p_test)
    print("R2 score: " + str(r2_score(y_test, p_test)))
    res = pd.DataFrame({'Actual': y_test, 'Predicted': p_test})
    pred = pd.DataFrame(data=p_test)
    pred.to_csv('pred.csv')


def gbt(X, y, X_train, X_test, y_train, y_test):
    gb_model = GradientBoostingRegressor(loss='ls', learning_rate=0.7)
    gb_model.fit(X_train, y_train)
    p_train = gb_model.predict(X_train)
    p_test = gb_model.predict(X_test)
    mae_train = mean_absolute_error(y_train, p_train)
    mae_test = mean_absolute_error(y_test, p_test)
    print(cross_val_score(GradientBoostingRegressor(loss='ls', learning_rate=0.7), X, y))
    print("R2 score: " + str(r2_score(y_test, p_test)))
    res = pd.DataFrame({'Actual': y_test, 'Predicted': p_test})
    pred = pd.DataFrame(data=p_test)
    pred.to_csv('pred.csv')
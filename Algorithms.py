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
    sns.heatmap(data.corr(), cbar=True, square=False, yticklabels=data.columns, xticklabels=data.columns)
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

def rf(X_train, X_test, y_train, y_test, models):
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rf_model.fit(X_train, y_train)
    print("R2 score: " + str(r2_score(y_test, rf_model.predict(X_test))))
    models.append(('rf', rf_model))


def dt(X_train, X_test, y_train, y_test, models):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    print("R2 score: " + str(r2_score(y_test, dt_model.predict(X_test))))
    models.append(('dt', dt_model))


def ada(X_train, X_test, y_train, y_test, models):
    ada_model = AdaBoostRegressor()
    ada_model.fit(X_train, y_train)
    print("R2 score: " + str(r2_score(y_test, ada_model.predict(X_test))))
    models.append(('ada', ada_model))


def gbt(X, y, X_train, X_test, y_train, y_test, models):
    gb_model = GradientBoostingRegressor(loss='ls', learning_rate=0.7)
    gb_model.fit(X_train, y_train)
    #print(cross_val_score(GradientBoostingRegressor(loss='ls', learning_rate=0.7), X, y))
    print("R2 score: " + str(r2_score(y_test, gb_model.predict(X_test))))
    models.append(('gbt', gb_model))

def compare(models, X_test, y_test):
    best=0
    for name, model in models:
        p_test=model.predict(X_test)
        score = r2_score(y_test, p_test)
        if(score > best):
            best=score
            best_name=name
            best_test=p_test
    return best, best_name, best_test
        


    
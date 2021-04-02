import pandas as pd
import seaborn as sns
import sklearn
import sys
import re
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler 
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor

def heatmap(data):
    data.corr()  # correlazione tra variabili in tabella n * n
    sns.heatmap(data.corr(), cbar=True, square=False, yticklabels=data.columns, xticklabels=data.columns)
    plt.show()
    data.head()

def scaledata(data):
    new= data["detection_time"].str.split(" ", n = 0, expand = True)
    data['date'] = new[0]
    data['time'] = new[1]
    data['time'] = pd.Series((data['time']).astype(str).str.replace(':', '', regex=False))
    data['date'] = pd.Series((data['date']).astype(str).str.replace('-', '', regex=False))
    data['time'] = data['time'].astype(int)
    data['date'] = data['date'].astype(int)
    data = data.drop(["detection_time","tracked_point_id"], axis=1)
    data['holiday'] = data.holiday.astype(int)
    transformers = [
        ['one_hot', OneHotEncoder(), ['season','weather', 'time']],
        ['scaler', StandardScaler(), ['weather_index','event_index','attractions_index','time_index', 'date']],
    ]
    ct = ColumnTransformer(transformers, remainder="passthrough")
    X = ct.fit_transform(data)
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        print(data.head())
    X = data.drop(["people_concentration"], axis=1)  # features
    y = data["people_concentration"]  # labels
    #heatmap(data)
    return X,y

def rf(X_train, X_test, y_train, y_test, models):
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=20000, random_state=0, min_samples_leaf=50)
    rf_model.fit(X_train, y_train)
    print("Random Forest score: " + str(r2_score(y_test, rf_model.predict(X_test))))
    models.append(('rf', rf_model))

def dt(X_train, X_test, y_train, y_test, models):
    dt_model = DecisionTreeRegressor(max_depth=None, min_samples_split=30000, min_samples_leaf=750, min_weight_fraction_leaf=0.0)
    dt_model.fit(X_train, y_train)
    print("Decision Tree score: " + str(r2_score(y_test, dt_model.predict(X_test))))
    models.append(('dt', dt_model))

def ada(X_train, X_test, y_train, y_test, models):
    ada_model = AdaBoostRegressor(learning_rate=0.1, n_estimators=50, loss='exponential')
    ada_model.fit(X_train, y_train)
    print("ADA Boost score: " + str(r2_score(y_test, ada_model.predict(X_test))))
    models.append(('ada', ada_model))

def gbt(X, y, X_train, X_test, y_train, y_test, models):
    gb_model = GradientBoostingRegressor(n_estimators=50, subsample=0.8, loss='ls', learning_rate=0.1, min_samples_split=10000, min_samples_leaf=50, max_depth=None)
    gb_model.fit(X_train, y_train)
    print("Gradient Boost score: " + str(r2_score(y_test, gb_model.predict(X_test))))
    models.append(('gbt', gb_model))

def xgb(X, y, X_train, X_test, y_train, y_test, models):
    xgb_model = XGBRegressor(learning_rate=0.1, max_depth=2, gamma=10)
    xgb_model.fit(X_train, y_train)
    print("XGBoost score: " + str(r2_score(y_test, xgb_model.predict(X_test))))
    models.append(('xgb', xgb_model))

#def lgbm(X, y, X_train, X_test, y_train, y_test, models):
#    lgbm_model = LGBMRegressor(learning_rate=0.1, max_depth=8, boosting_type="goss", max_bin=25, num_leaves=5, path_smooth=1.0, min_data_in_leaf=1000, num_iterations=100)
#    lgbm_model.fit(X_train, y_train)
#    print("Light GBM score: " + str(r2_score(y_test, lgbm_model.predict(X_test))))
#    models.append(('lgbm', lgbm_model))

def compare(models, X_test, y_test):
    best=0 
    best_name=None
    best_test=0
    for name, model in models:
        p_test=model.predict(X_test)
        score = r2_score(y_test, p_test)
        if score > best:
            best=score
            best_name=name
            best_test=model
    return best, best_name, best_test
        


    

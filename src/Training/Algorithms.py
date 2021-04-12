import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler 
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

def heatmap(data):
    data.corr() 
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
    x = ct.fit_transform(data)
    x = data.drop(["people_concentration"], axis=1)  # features
    y = data["people_concentration"]  # labels
    return x,y

def rf(x_train, x_test, y_train, y_test, models):
    rf_model = RandomForestRegressor(bootstrap = True, max_depth = 10, max_features = 'log2', min_samples_leaf = 5, min_samples_split = 5, n_estimators = 100, random_state = None)
    rf_model.fit(x_train, y_train)
    print("Random Forest score: " + str(r2_score(y_test, rf_model.predict(x_test))))
    models.append(('rf', rf_model))

def dt(x_train, x_test, y_train, y_test, models):
    dt_model = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 10, min_samples_split = 500, min_weight_fraction_leaf = 0.0, random_state = None, splitter = 'random', max_features = 'auto')
    dt_model.fit(x_train, y_train)
    print("Decision Tree score: " + str(r2_score(y_test, dt_model.predict(x_test))))
    models.append(('dt', dt_model))

def gbt(x_train, x_test, y_train, y_test, models):
    gb_model = GradientBoostingRegressor(learning_rate= 0.01, loss= 'ls', max_depth= 8, n_estimators= 150, subsample= 0.2)
    gb_model.fit(x_train, y_train)
    print("Gradient Boost score: " + str(r2_score(y_test, gb_model.predict(x_test))))
    models.append(('gbt', gb_model))

def compare(models, x_test, y_test):
    best=0 
    best_name=None
    best_test=0
    for name, model in models:
        p_test=model.predict(x_test)
        score = r2_score(y_test, p_test)
        if score > best:
            best=score
            best_name=name
            best_test=model
    return best, best_name, best_test
        


    

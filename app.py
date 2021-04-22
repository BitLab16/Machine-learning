from flask import Flask, request, jsonify
from flask_injector import FlaskInjector
from injector import inject
from injector import singleton
from sklearn.metrics import r2_score

from datetime import datetime

from flaskr.ml import Algorithm, DecisionTree, GradientBoosting, RandomForest
from flaskr.ml import DecisionTreeFactory
from flaskr.ml import GradientBoostingFactory
from flaskr.ml import RandomForestFactory
from flaskr.ml import Scaler
from flaskr.ml import Trainer
from flaskr.service import PredictionService
from flaskr.repository import FileReader
from flaskr.repository import Db


app = Flask(__name__)


def compare_algorithm(models, x_test, y_test):
    best=0 
    best_test=0
    for model in models:
        p_test=model.predict(x_test)
        score = r2_score(y_test, p_test)
        if score > best:
            best=score
            best_test=model
    return best, best_test


def to_date(date_string): 
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M")


def train_ml():
    file_reader = FileReader("train.csv")
    data = file_reader.read_file()
    decision_tree = DecisionTreeFactory().create()
    gradient_boosting = GradientBoostingFactory().create()
    random_forest = RandomForestFactory().create()
    models = [decision_tree, gradient_boosting, random_forest]
    scaler = Scaler()
    feature, targets = scaler.scale(data)
    trainer = Trainer(0.2, feature, targets)
    trainer.train(models)
    score, model = compare_algorithm(models, trainer.x_test, trainer.y_test)
    print("Best score: " + str(score))
    return model


def configure(binder):
    model = train_ml()
    binder.bind(PredictionService, to=PredictionService, scope=singleton)
    binder.bind(Algorithm, to=model, scope=singleton)
    binder.bind(Db, to=Db, scope=singleton)
    binder.bind(FileReader, to=FileReader('predict.csv'), scope=singleton)
    binder.bind(Scaler, to=Scaler, scope=singleton)


@app.route('/prediction/')
def prediction_for_all_point_in_interval(service: PredictionService):
    time_from = request.args.get('from', default = '', type = to_date)
    time_to = request.args.get('to', default = '', type = to_date)
    tracked_point_list = service.predict(time_from, time_to)
    result = []
    i = 0
    for item in tracked_point_list:
        result.append({"point_id":int(item.point_id),  "predictions": []})
        for predict in item.predictions:
            result[i]["predictions"].append({str(predict.time): predict.flow})
        i = i+1
    return jsonify(result)


@app.route('/prediction/<int:code>/')
def prediction_for_single_point_in_interval(code: int, service: PredictionService):
    time_from = request.args.get('from', default = '', type = to_date)
    time_to = request.args.get('to', default = '', type = to_date)
    tracked_point = service.predict_for_point(code, time_from, time_to)
    result = { "point_id": tracked_point.point_id, "predictions": []}
    for item in tracked_point.predictions:
        result["predictions"].append({str(item.time): item.flow})
    return jsonify(result)


FlaskInjector(app=app, modules=[configure])
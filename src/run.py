import schedule
import time
from Prediction.predict import predictions
from Training.traincode import traincode
X, y, models, data, best, best_name, best_test, engine = traincode()
schedule.every(5).seconds.do(predictions,X, y, models, data, best, best_name, best_test, engine)
while True:
    schedule.run_pending()
    time.sleep(1)

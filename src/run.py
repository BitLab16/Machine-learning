import schedule
import time
import Prediction.predict
import Training.traincode
X, y, models, data, best, best_name, best_test, engine = Training.traincode.traincode()
schedule.every(5).seconds.do(Prediction.predict.predictions,X, y, models, data, best, best_name, best_test, engine)
while True:
    schedule.run_pending()
    time.sleep(1)
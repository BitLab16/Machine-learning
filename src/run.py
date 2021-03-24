import schedule
import time
import Prediction.predict
import Training.traincode
X, y, models, data, engine=Training.traincode.traincode()
schedule.every(15).seconds.do(Prediction.predict.predictions,X, y, models, data, engine)
while True:
    schedule.run_pending()
    time.sleep(1)
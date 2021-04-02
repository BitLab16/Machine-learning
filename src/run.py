import schedule
import time
import predict
import traincode
X, y, models, data, best, best_name, best_test, engine = traincode.traincode()
schedule.every(5).seconds.do(predict.predictions,X, y, models, data, best, best_name, best_test, engine)
while True:
    schedule.run_pending()
    time.sleep(1)
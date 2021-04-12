import schedule
import time
from Prediction import predict
from Training import traincode
x, y, models, data, best, best_name, best_test, engine = traincode.traincode()
predict.predictions(x, y, models, data, best, best_name, best_test, engine)
#schedule.every(5).seconds.do(predict.predictions, x, y, models, data, best, best_name, best_test, engine)
#while True:
#    schedule.run_pending()
#    time.sleep(1)
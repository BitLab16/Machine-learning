import schedule
import train
import time

schedule.every(15).seconds.do(train.traincode)
while True:
    schedule.run_pending()
    time.sleep(1)
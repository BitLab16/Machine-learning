from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from sqlalchemy import insert, update
import sql

def connect():
    engine = create_engine('postgresql+psycopg2://user:user@localhost:6543/gathering_detection')
    connection = engine.connect()
    print(engine.table_names())
    metadata = MetaData()
    gatherings_detection = Table('gatherings_detection', metadata, autoload=True, autoload_with=engine)
    gatherings_prediction = Table('gatherings_prediction', metadata, autoload=True, autoload_with=engine)
    return engine, gatherings_detection, gatherings_prediction, connection
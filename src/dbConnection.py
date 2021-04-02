import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from geoalchemy2 import Geography
# import sql
import re
from sqlalchemy import insert, update
import psycopg2

def connect():
    engine = create_engine('postgresql+psycopg2://user:user@postgres-db:5432/gathering_detection')
    connection = engine.connect()
    print(engine.table_names())
    metadata = MetaData()
    gatherings_detection = Table('gatherings_detection', metadata, autoload=True, autoload_with=engine)
    gatherings_prediction = Table('gatherings_prediction', metadata, autoload=True, autoload_with=engine)
    return engine, gatherings_detection, gatherings_prediction, connection

def getTables(connection):
    data = pd.read_sql_table('gatherings_detection', con=connection)
    prediction_df = pd.read_sql_table('gatherings_prediction', con=connection)
    return data, prediction_df


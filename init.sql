-- Script per inizializzazione db



CREATE TABLE IF NOT EXISTS tracked_point (
    id int PRIMARY KEY NOT NULL,
    code int UNIQUE NOT NULL,
    point_name varchar(50) NOT NULL, 
    description text NULL,
    location varchar(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS gatherings_detection(
    id int PRIMARY KEY NOT NULL,
    tracked_point_id int NOT NULL,
    detection_time TIMESTAMP NOT NULL,
    season int NOT NULL,
    holiday BOOLEAN NOT NULL,
    time_index FLOAT NOT NULL,
    weather_index FLOAT NOT NULL,
    season_index FLOAT NOT NULL,
    attractions_index FLOAT NOT NULL,
    people_concentration int NOT NULL,
    FOREIGN KEY (tracked_point_id) REFERENCES tracked_point (id)
);

CREATE TABLE IF NOT EXISTS gatherings_prediction(
    id int PRIMARY KEY NOT NULL,
    tracked_point_id int NOT NULL,
    detection_time TIMESTAMP NOT NULL,
    people_concentration int NOT NULL,
    FOREIGN KEY (tracked_point_id) REFERENCES tracked_point (id)
);

INSERT INTO tracked_point (id,code,point_name,description,location) VALUES ('001','001','001','boh','Padova')
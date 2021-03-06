FROM python:3.8-slim-buster

ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# COPY ./requirements.txt /requirements.txt

# WORKDIR /

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /

RUN python setup.py install

CMD ["flask", "run", "-h", "0.0.0.0", "-p", "6001"]

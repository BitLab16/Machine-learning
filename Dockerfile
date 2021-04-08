FROM python:3.8-slim-buster

ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PYTHONPATH="$PYTHONPATH:/src"

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY /src .

CMD ["python", "run.py"]

FROM python:3.9
USER root
WORKDIR /tmp/work

ENV TZ JST-9

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

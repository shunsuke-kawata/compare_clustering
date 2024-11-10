FROM python:3.9
USER root
WORKDIR /tmp/work

ENV TZ JST-9

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
RUN apt -y update && apt -y upgrade
RUN apt -y install libopencv-dev libhdf5-dev

COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt
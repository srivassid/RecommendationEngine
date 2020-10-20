FROM ubuntu:latest
MAINTAINER <siddharthas@connecterra.io>
FROM python:3.8
RUN apt-get update
COPY requirements.txt /tmp

WORKDIR /tmp
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

WORKDIR /home/GetNewMovies

COPY send_data_topic.py /home/GetNewMovies/
RUN pwd
RUN ls
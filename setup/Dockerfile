FROM python:3.12-alpine3.21

LABEL maintainer="Alan Hsieh"

RUN apk update

RUN apk add git

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    gymnasium

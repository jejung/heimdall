FROM python:2.7

ENV PYTHONUNBUFFERED 1
ENV DEBUG 1

WORKDIR /var/www/heimdall.dev/

ADD requirements.txt /var/www/heimdall.dev/

RUN pip install -r requirements.txt

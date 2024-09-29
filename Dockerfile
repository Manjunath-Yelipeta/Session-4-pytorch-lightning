FROM python:3.9-slim-bullseye


WORKDIR /opt

RUN mkdir -p /opt/logs
RUN mkdir -p /opt/data
RUN mkdir -p /opt/exp
RUN mkdir -p /opt/ext
RUN mkdir -p /opt/samples
RUN mkdir -p /opt/samples
RUN mkdir -p /opt/predictions


COPY requirements.txt requirements.txt



RUN pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip
#COPY train.py /opt/src
COPY . .

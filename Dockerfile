FROM python:3.8.6-slim

ADD requirements.txt .
RUN pip install -r requirements.txt

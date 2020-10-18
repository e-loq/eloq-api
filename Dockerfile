FROM python:3.8.6-slim

ADD ./requirements.txt /src/
ADD ./Explore_Data.py /src/
ADD data/df_reduced.pkl src/data/

WORKDIR /src/
RUN pip install -r requirements.txt

EXPOSE 5000

CMD [ "python", "./api.py" ]

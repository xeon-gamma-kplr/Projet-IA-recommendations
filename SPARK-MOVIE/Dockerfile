FROM apache/spark:3.4.0

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

COPY ./app /app

COPY ./requirements.txt /app

COPY ./app/csv /csv

COPY ./app/templates /templates

RUN pip3 install -r requirements.txt

EXPOSE 5432

CMD ["spark-submit","server.py","csv/movies.csv","csv/ratings.csv"]
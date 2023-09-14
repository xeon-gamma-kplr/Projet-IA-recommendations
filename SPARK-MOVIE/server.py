import time,sys,os,cherrypy,cheroot.wsgi
from pyspark import SparkConf, SparkContext
from app import create_app

conf = SparkConf().setAppName("movie_recommendation-server")

sc = SparkContext(conf=conf, pyFiles=['SPARK-MOVIE/engine.py', 'SPARK-MOVIE/app.py'])

movies_set_path = "/workspace/Projet-IA-recommendations/SPARK-MOVIE/parquets/movies.parquet"
ratings_set_path = "/workspace/Projet-IA-recommendations/SPARK-MOVIE/parquets/ratings.parquet"

app = create_app(sc,movies_set_path,ratings_set_path)

app.run(debug=True, host='0.0.0.0', port=5066)

# cherrypy.tree.graft(app.wsgi_app, '/') 
# cherrypy.config.update({'server.socket_host': '0.0.0.0','server.socket_port': 5432,'engine.autoreload.on': False})
# cherrypy.engine.start()
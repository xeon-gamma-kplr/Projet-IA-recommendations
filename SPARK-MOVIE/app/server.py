import time,sys,os,cherrypy,cheroot.wsgi
from pyspark import SparkConf, SparkContext
from app import create_app

current_dir = os.path.realpath(os.path.dirname(__file__))

conf = SparkConf().setAppName("movie_recommendation-server")

sc = SparkContext(conf=conf, pyFiles=[current_dir + '/engine.py', current_dir +'/app.py'])


csv_path = [current_dir +"/csv" + f'/{file}' for file in os.listdir(current_dir + "/csv")]
parquet_path = [current_dir + '/parquet' + f'/{file}' for file in ("movies.parquet","ratings.parquet")]
model_path = current_dir + '/model/als_model'



app = create_app(sc,csv_path,parquet_path,model_path)

# app.run(debug=True, host='0.0.0.0', port=5066)

cherrypy.tree.graft(app.wsgi_app, '/') 
cherrypy.config.update({'server.socket_host': '0.0.0.0','server.socket_port': 5432,'engine.autoreload.on': False})
cherrypy.engine.start()
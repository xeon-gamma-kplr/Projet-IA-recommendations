from pyspark.sql.types import StringType, ArrayType,StructType,StructField,IntegerType,FloatType
from pyspark.sql.functions import split, regexp_replace, explode, col,avg, count,max
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
import numpy as np
import random

def create_spark_session():
    sc = SparkContext('local')
    return SparkSession(sc)


def create_parquet_files():
    schema = StructType([ 
        StructField("movieID",IntegerType(),True), 
        StructField("title",StringType()), 
        StructField("genres",StringType())
    ])

    # df_movies = spark.read.option("header","true").option("inferSchema","true").csv("dbfs:/FileStore/movies.csv")
    df_movies = spark.read.option("header","true").schema(schema).csv("/workspace/Projet-IA-recommendations/csv/movies.csv")
    # df_movies = df_movies.withColumn("Genres", regexp_replace('Genres', "|",","))
    df_movies.write.mode("overwrite").parquet("/workspace/Projet-IA-recommendations/parquets/movies.parquet")
    #df_movies = spark.read.parquet("/workspace/Projet-IA-recommendations/parquets/movies.parquet")

    schema = StructType([ 
        StructField("userID",IntegerType(),True), 
        StructField("movieID",IntegerType()), 
        StructField("rating",FloatType()),
        StructField("timestamp",IntegerType())
    ])

    df_ratings = spark.read.option("header","true").schema(schema).csv("/workspace/Projet-IA-recommendations/csv/ratings.csv")
    df_ratings.write.mode("overwrite").parquet("/workspace/Projet-IA-recommendations/parquets/ratings.parquet")
    # df_ratings = spark.read.parquet("/workspace/Projet-IA-recommendations/parquets/ratings.parquet")

def load_parquet_file(path):
    try:
        df = spark.read.parquet(path)
    except:
        create_parquet_files()
        df = spark.read.parquet(path)
    return df

def create_model(path,datas):
    try:
        model = load_model(path)
        datas.all_recommendations = model.recommendForAllUsers(5)
        print("load model...")
    except:
        model = train_model(datas)
    return model

def train_model(datas):
    print("train model...")
    df = datas.ratings.drop("timestamp")
    als = ALS(maxIter=5, rank=4, regParam=0.01, userCol='userID', itemCol='movieID', ratingCol='rating', coldStartStrategy='drop')
    model = als.fit(df)
    datas.all_recommendations = model.recommendForAllUsers(5)
    save_model(model)
    return model

def save_model(model):
    print("save model...")
    model.write().overwrite().save("/workspace/Projet-IA-recommendations/model/als_model")

def load_model(path):
    return ALSModel.load(path)

def find_movie_name(movie_id, movie_df):
    title = movie_df.where(movie_df.movieID == movie_id).collect()[0]["title"]
    return title

class User:
    def __init__(self, id,datas):
        self.id = id
        self.ratings = []
        self.recommendations = []
        self.get_recommendations(model,datas.movies)
    def get_recommendations(self, model, df_movies):
        self.recommendations = [find_movie_name(movie['movieID'], df_movies) for movie in datas.all_recommendations.where(datas.all_recommendations["userID"] == self.id).collect()[0]['recommendations']]

class Data:
    def __init__(self, ratings_path, movie_path):
        self.ratings = load_parquet_file(ratings_path)
        self.movies = load_parquet_file(movie_path)
        self.all_recommendations = []

def load_user(datas):
    user_id = int(input("user_id: "))
    if datas.ratings.filter(datas.ratings["userID"].isin([user_id])).collect() == []:
        add_user(user_id, datas)
    return User(user_id,datas)

def add_user(user_id, datas):
    movie_list = []
    for i in range(5):
        while True:
            movie_id = random.choice(datas.ratings.dropDuplicates(["movieID"]).select("movieID").collect())
            if movie_id not in movie_list:
                movie_list.append(movie_id["movieID"])
                break
    ranking_list = []
    for movie in movie_list:
        while True:
            print(find_movie_name(movie, datas.movies))
            rating = input("note sur 5: ")
            try:
                rating = float(rating)
                ranking_list.append((user_id,movie,rating,-1))
                break
            except:
                "erreur"
    new_user = spark.createDataFrame(ranking_list, datas.ratings.columns)
    datas.ratings = datas.ratings.union(new_user).distinct()
    model = train_model(datas)
    max_user_id(datas)
    datas.ratings.write.mode("overwrite").parquet("/workspace/Projet-IA-recommendations/parquets/ratings.parquet")

def max_user_id(datas):
    datas.ratings.select(max("userID")).show()

def add_rating(movie_id,datas):
    while True:
        print(find_movie_name(movie_id, datas.movies))
        rating = input("note sur 5: ")
        try:
            rating = float(rating)
            return (user_id,movie,rating,-1)
        except:
            print("erreur")



spark = create_spark_session()

datas = Data("/workspace/Projet-IA-recommendations/parquets/ratings.parquet","/workspace/Projet-IA-recommendations/parquets/movies.parquet")

# movie_df = load_parquet_file("/workspace/Projet-IA-recommendations/parquets/movies.parquet")

# main_df = load_parquet_file("/workspace/Projet-IA-recommendations/parquets/ratings.parquet")

model = create_model("/workspace/Projet-IA-recommendations/model/als_model", datas)

max_user_id(datas)

user = load_user(datas)

print(user.recommendations)


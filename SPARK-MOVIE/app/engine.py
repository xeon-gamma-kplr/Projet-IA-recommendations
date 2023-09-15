from pyspark.sql.types import StringType, ArrayType,StructType,StructField,IntegerType,FloatType
from pyspark.sql.functions import split, regexp_replace, explode, col,avg, count,max
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
import numpy as np
import random

# def create_spark_session():
#     sc = SparkContext('local')
#     return SparkSession(sc)


def create_parquet_files(spark,csv_path,parquet_path):
    schema = StructType([ 
        StructField("movieID",IntegerType(),True), 
        StructField("title",StringType()), 
        StructField("genres",StringType())
    ])

    # df_movies = spark.read.option("header","true").option("inferSchema","true").csv("dbfs:/FileStore/movies.csv")
    df_movies = spark.read.option("header","true").schema(schema).csv(csv_path[0])
    # df_movies = df_movies.withColumn("Genres", regexp_replace('Genres', "|",","))
    df_movies.write.mode("overwrite").parquet(parquet_path[0])
    #df_movies = spark.read.parquet("/workspace/Projet-IA-recommendations/parquets/movies.parquet")

    schema = StructType([ 
        StructField("userID",IntegerType(),True), 
        StructField("movieID",IntegerType()), 
        StructField("rating",FloatType()),
        StructField("timestamp",IntegerType())
    ])

    df_ratings = spark.read.option("header","true").schema(schema).csv(csv_path[1])
    df_ratings.write.mode("overwrite").parquet(parquet_path[1])
    # df_ratings = spark.read.parquet("/workspace/Projet-IA-recommendations/parquets/ratings.parquet")


def load_parquet_file(spark,parquet_path,csv_path):
    try:
        df_ratings = spark.read.parquet(parquet_path[1])
        df_movies = spark.read.parquet(parquet_path[0])
    except:
        create_parquet_files(spark,csv_path,parquet_path)
        df_ratings = spark.read.parquet(parquet_path[1])
        df_movies = spark.read.parquet(parquet_path[0])
    return df_ratings,df_movies

class Model:
    def __init__(self,spark,path,datas):
        self.path = path
        self.model = None
        self.create_model(datas,path)
    
    def create_model(self,datas,path):
        try:
            self.load_model(path,datas)
            print("load model...")
        except:
            self.train_model(datas)
    
    def recommandation(self,datas,num):
        datas.all_recommendations = self.model.recommendForAllUsers(num)


    def load_model(self,path,datas):
        self.model = ALSModel.load(path)
        self.train_model(datas)
    
    def train_model(self,datas):
        print("train model...")
        df = datas.ratings.drop("timestamp")
        als = ALS(maxIter=5, rank=4, regParam=0.01, userCol='userID', itemCol='movieID', ratingCol='rating', coldStartStrategy='drop')
        self.model = als.fit(df)
        # self.save_model(self.path)
        self.recommandation(datas,5)

    # def save_model(self,path):
    #     print("save model...")
    #     self.model.write().overwrite().save(path)

# def train_model(datas):
#     print("train model...")
#     df = datas.ratings.drop("timestamp")
#     als = ALS(maxIter=5, rank=4, regParam=0.01, userCol='userID', itemCol='movieID', ratingCol='rating', coldStartStrategy='drop')
#     model = als.fit(df)
#     datas.all_recommendations = model.recommendForAllUsers(5)
#     save_model(model)
#     return model

# def save_model(model):
#     print("save model...")
#     model.write().overwrite().save("/workspace/Projet-IA-recommendations/SPARK-MOVIE/model/als_model")

# def load_model(path):
#     return ALSModel.load(path)

def find_movie_name(movie_id, movie_df):
    title = movie_df.where(movie_df.movieID == movie_id).collect()[0]["title"]
    return title

class User:
    def __init__(self,id,datas):
        self.id = id
        # self.ratings = []
        # self.recommendations = []
        # self.get_recommendations(datas.movies,datas)
        # self.get_ratings(datas)
    def get_recommendations(self,df_movies,datas):
        return [find_movie_name(movie['movieID'], df_movies) for movie in datas.all_recommendations.where(datas.all_recommendations["userID"] == self.id).collect()[0]['recommendations']]
    def get_ratings(self,datas):
        return {find_movie_name(movie['movieID'],datas.movies) : movie["rating"] for movie in datas.ratings.select("movieID","rating").where(datas.ratings["userID"] == self.id).collect()}

class Data:
    def __init__(self, spark,csv_path, parquet_path):
        self.path = [csv_path,parquet_path]
        self.ratings,self.movies = load_parquet_file(spark,parquet_path,csv_path)
        self.all_recommendations = []
        self.cache(self.ratings)
    def cache(self,df):
        print("cache")
        df.cache()
    def save(self,spark):
        print("saving datas...")
        self.ratings.write.mode("overwrite").parquet(self.path[1][1])
        print("reload datas..")
        self.ratings = load_parquet_file(spark,self.path[1],self.path[0])
        self.cache(self.ratings)

# def load_user(model,spark,datas):
#     user_id = int(input("user_id: "))
#     if datas.ratings.filter(datas.ratings["userID"].isin([user_id])).collect() == []:
#         add_user(model,spark,user_id, datas)
#     return User(user_id,datas)

def is_user_known(user_id,datas):
    print(user_id)
    print(datas.ratings.filter(datas.ratings["userID"].isin([user_id])).collect())
    if datas.ratings.filter(datas.ratings["userID"].isin([user_id])).collect() == []:
        return False
    else:
        return True

def add_user(model,spark,user_id, datas):
    movie_list = []
    for i in range(5):
        while True:
            movie_id = random.choice(datas.ratings.dropDuplicates(["movieID"]).select("movieID").collect())
            if movie_id not in movie_list:
                movie_list.append(movie_id["movieID"])
                break
    ranking_list = []
    for movie in movie_list:
        ranking_list.append(select_rating(user_id,movie,datas))
    add_row(model,spark,ranking_list, datas)

def add_row(model,spark,ranking_list,datas):
    print(ranking_list)
    new_user = spark.createDataFrame(ranking_list, datas.ratings.columns)
    datas.ratings = datas.ratings.union(new_user).distinct()
    # max_user_id(datas)
    model.train_model(datas)
    datas.save(spark)

def max_user_id(datas):
    return(datas.ratings.select(max("userID")).collect()[0]["max(userID)"])

def add_rating(user_id,dico,datas,model,spark):
    dico_values = list(dico.values())   
    rating = [(user_id,int(e[0]),int(e[1]),-1) for e in zip(dico_values[0::2],dico_values[1::2])]
    # rating = [select_rating(user_id,movie,datas)]
    add_row(model,spark,rating,datas)

def select_rating(user_id,movie_id,datas):
    while True:
        print(find_movie_name(movie_id, datas.movies))
        rating = input("note sur 5: ")
        try:
            rating = float(rating)
            return (user_id,movie_id,rating,-1)
        except:
            print("erreur")

def movie_prediction_for_user(user_id,movie_id,model):
    users_factors = model.userFactors

    item_factors = model.itemFactors

    return (np.matmul(np.array(users_factors.where(users_factors.id == user_id).collect()[0]["features"]).reshape(1,4),(np.array(item_factors.where(item_factors.id == movie_id).collect()[0]["features"]).reshape(4,1))))


if __name__ == "__main__":
    pass
    


    # while True:
    #     datas = Data(spark,"/workspace/Projet-IA-recommendations/SPARK-MOVIE/parquets/ratings.parquet","/workspace/Projet-IA-recommendations/SPARK-MOVIE/parquets/movies.parquet")
    #     model = Model(spark,"/workspace/Projet-IA-recommendations/SPARK-MOVIE/model/als_model", datas)
    #     max_user_id(datas)
    #     user = load_user(model,spark,datas)
    #     print(user.recommendations)
    #     choice = input("voulez vous ajouter un film: ")
    #     if choice == "oui":
    #         add_rating(user.id,datas,model,spark)


import flask,json,findspark
from engine import *
from pyspark.sql.session import SparkSession

main = flask.Blueprint('main', __name__)
findspark.init()

def create_app(spark_context, movies_set_path, ratings_set_path):
    global spark,datas,model
    # Code pour initialiser le moteur de recommandation avec le contexte Spark et les jeux de données
    spark = SparkSession(spark_context)
    spark.conf.set("spark.sql.parquet.enableVectorizedReader","false")
    datas = Data(spark,ratings_set_path,movies_set_path)
    model = Model(spark,"/workspace/Projet-IA-recommendations/SPARK-MOVIE/model/als_model", datas)
    # Créez une instance de l'application Flask
    app = flask.Flask(__name__)
    # Enregistrez le Blueprint "main" dans l'application
    # Configurez les options de l'application Flask
    app.register_blueprint(main)
    # Renvoyez l'application Flask créées
    return app
    

@main.route("/", methods=["GET", "POST", "PUT"])
def home():
    return flask.render_template("index.html", variable=max_user_id(datas))

@main.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    return {movie_id: find_movie_name(movie_id, datas.movies)}
    # Code pour récupérer les détails du film avec l'id spécifié
    # et renvoyer les données au format JSON

@main.route("/newratings/<int:user_id>/<int:number>", methods=["POST","GET"])
def new_ratings(user_id,number):
    if flask.request.method == "GET":
        return flask.render_template("add_rating.html",number=int(number))
    elif flask.request.method == "POST":
        exist = is_user_known(user_id,datas)
        # add_rating(int(user_id),int(flask.request.form["movieid"]),float(flask.request.form["rating"]),datas,model,spark)
        add_rating(user_id,dict(flask.request.form),datas,model,spark)
        if exist:
            return str("")
        else:
            return str(user_id)

@main.route("/<int:user_id>/ratings/<int:movie_id>", methods=["GET"])
def movie_ratings(user_id, movie_id):
    return str(movie_prediction_for_user(user_id,movie_id,model.model)[0][0])

@main.route("/recommend/<int:user_id>", methods=["GET"])
def recommend_for_user(user_id):
    # return str(datas.all_recommendations)
    user = User(user_id,datas)
    return flask.render_template("recommendations.html",liste=user.get_recommendations(datas.movies,datas))

@main.route("/ratings/<int:user_id>", methods=["GET"])
def get_ratings_for_user(user_id):
    user=User(user_id,datas)
    return user.get_ratings(datas)

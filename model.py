from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import os.path
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier

if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)

    path_dataset = './filteredC.small.training'

    sqlc = SQLContext(sc)
    df = sqlc.read.csv(path_dataset, header=True, sep=",", inferSchema=True)

    print(df)

    sc.stop()

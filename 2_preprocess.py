from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.feature import StringIndexer


def undersampling_first_df(df_0, df_1):
    ratio = 1.0*df_1.count()/df_0.count()
    df_0 = df_0.sample(False, ratio)
    return df_1.unionAll(df_0)


def undersampling(df):
    df_1 = df.filter(col("class") == 1)
    df_0 = df.filter(col("class") == 0)

    if df_1 > df_0:
        return undersampling_first_df(df_1, df_0)
    else:
        return undersampling_first_df(df_0, df_1)


def categorical_column_to_int(df, col_name):
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name)
    return indexer.fit(df).transform(df)


def preprocess_dataset(df):
    df = undersampling(df)
    categorical_column_to_int(df)
    df.write.csv('./preprocess', header=True, mode="overwrite")


if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)

    path_dataset = './filteredC.small.training'

    sqlc = SQLContext(sc)
    df = sqlc.read.csv(path_dataset, header=True, sep=",", inferSchema=True)

    preprocess_dataset(df)

    sc.stop()

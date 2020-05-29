from pyspark import SparkContext, SparkConf, SQLContext

if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)

    path_dataset = './filteredC.small.training'

    sqlc = SQLContext(sc)
    df = sqlc.read.csv(path_dataset, header=True, sep=",", inferSchema=True)

    print(df)

    sc.stop()

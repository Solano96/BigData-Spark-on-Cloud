from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC


def undersampling_first_df(df_0, df_1):
    ratio = 1.0*df_1.count()/df_0.count()
    df_0 = df_0.sample(False, ratio)
    return df_1.unionAll(df_0)


def undersampling(df):
    df_1 = df.filter(col("class") == 1)
    df_0 = df.filter(col("class") == 0)

    if df_1.count() > df_0.count():
        return undersampling_first_df(df_1, df_0)
    else:
        return undersampling_first_df(df_0, df_1)


def categorical_column_to_int(df, col_name):
    indexer = StringIndexer(inputCol=col_name, outputCol='new'+col_name)
    df = indexer.fit(df).transform(df)

    df = df.drop(col_name)
    df = df.withColumnRenamed('new'+col_name, col_name)

    return df


def scale_dataset(df_train, df_test):
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)
    scalerModel = scaler.fit(df_train)

    df_train_scaled = scalerModel.transform(df_train)
    df_test_scaled = scalerModel.transform(df_test)

    return df_train_scaled, df_test_scaled

def stratified_train_test_split(df):
    df_train = df.sampleBy('class', fractions={0:0.7, 1:0.7}, seed=10)
    df_test = df.subtract(df_train)
    return df_train, df_test


def preprocess_dataset(df, columns):
    df = categorical_column_to_int(df, 'PredSS_central')
    df = undersampling(df)

    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    df = assembler.transform(df).select("features", "class")

    df_train, df_test = stratified_train_test_split(df)
    df_train_scaled, df_test_scaled = scale_dataset(df_train, df_test)

    return df_train_scaled, df_test_scaled


if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)

    path_dataset = './filteredC.small.training'

    sqlc = SQLContext(sc)
    df = sqlc.read.csv(path_dataset, header=True, sep=",", inferSchema=True)

    preprocess_dataset(df)

    sc.stop()

if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)

    path_dataset = './filteredC.small.training'
    columns = ['PSSM_central_1_Y', 'PSSM_r2_4_T', 'PSSM_r1_-3_T', 'PSSM_r1_2_T', 'PSSM_r1_0_I', 'PredSS_central']

    sqlc = SQLContext(sc)
    df = sqlc.read.csv(path_dataset, header=True, sep=",", inferSchema=True)

    df_train, df_test = preprocess_dataset(df, columns)

    rf = RandomForestClassifier(
        featuresCol="features", labelCol="class", numTrees=15, impurity='gini', seed=89)
    model = rf.fit(df_train)
    evaluator = BinaryClassificationEvaluator()
    df_test = df_test.withColumnRenamed('class', 'label')
    evaluation = evaluator.evaluate(model.transform(df_test))
    print("[TEST] Area bajo la curva ROC:", evaluation)

    sc.stop()

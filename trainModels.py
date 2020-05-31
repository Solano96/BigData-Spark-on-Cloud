from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

def undersampling_major_class(df_minor, df_major):
    ratio = 1.0*df_minor.count()/df_major.count()
    df_major = df_major.sample(withReplacement=False, fraction=ratio)
    return df_minor.unionAll(df_major)


def undersampling(df):
    df_0 = df.filter(col("class") == 0)
    df_1 = df.filter(col("class") == 1)

    if df_0.count() < df_1.count():
        return undersampling_major_class(df_0, df_1)
    else:
        return undersampling_major_class(df_1, df_0)


def oversampling_minor_class(df_minor, df_major):
    ratio = 1.0*df_major.count()/df_minor.count()
    df_minor = df_minor.sample(withReplacement=True, fraction=ratio)
    return df_minor.unionAll(df_major)


def oversampling(df):
    df_0 = df.filter(col("class") == 0)
    df_1 = df.filter(col("class") == 1)

    if df_0.count() < df_1.count():
        return oversampling_minor_class(df_0, df_1)
    else:
        return oversampling_minor_class(df_1, df_0)


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


def preprocess_dataset(df, columns, balance_type = 'rus'):
    """ Preprocess dataset"""
    df = categorical_column_to_int(df, 'PredSS_central')

    df.groupBy('class').count().show()

    # Balance class
    if balance_type == 'rus':
        df = undersampling(df)
    elif balance_type == 'ros':
        df = oversampling(df)

    df.groupBy('class').count().show()

    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    df = assembler.transform(df).select("features", "class")

    df_train, df_test = stratified_train_test_split(df)
    df_train_scaled, df_test_scaled = scale_dataset(df_train, df_test)

    return df_train_scaled, df_test_scaled


def train_random_forest(df_train, numTrees = 100, maxDepth = 5, impurity = "gini"):
    """ Train a random forest model"""
    rf = RandomForestClassifier(featuresCol="scaledFeatures",
                                labelCol="class",
                                numTrees=numTrees,
                                maxDepth=maxDepth,
                                impurity=impurity)

    rf_model = rf.fit(df_train)

    str_params = 'numTrees: ' + str(numTrees)
    str_params += ', maxDepth: ' + str(maxDepth)
    str_params += ', impurity: ' + str(impurity)

    return rf_model, str_params


def train_logistic_regression(df_train, maxIter=100, regParam=0.0):
    """ Train a logistic regression model"""
    lr = LogisticRegression(featuresCol="scaledFeatures",
                            labelCol="class",
                            maxIter=maxIter,
                            regParam=regParam)

    lr_model = lr.fit(df_train)

    str_params = 'maxIter: ' + str(maxIter)
    str_params += ', regParam: ' + str(regParam)

    return lr_model, str_params


def train_decision_tree(df_train, maxDepth = 5, impurity = "gini"):
    """ Train a decision tree model"""
    dt = DecisionTreeClassifier(featuresCol="scaledFeatures",
                                labelCol="class",
                                numTrees=numTrees,
                                maxDepth=maxDepth,
                                impurity=impurity)

    dt_model = dt.fit(df_train)

    str_params = 'maxDepth: ' + str(maxDepth)
    str_params += ', impurity: ' + str(impurity)

    return dt_model, str_params


def evaluate(model, df_test, clf_name, clf_params):
    """ Evaluate model output and print result """

    print("Model -> " + clf_name)
    print("Params -> " + clf_params)

    evaluator = BinaryClassificationEvaluator(labelCol="class")
    predictions = model.transform(df_test)
    auc = evaluator.evaluate(predictions)

    predictionAndLabels = predictions.select('prediction', 'class').rdd
    predictionAndLabels = predictionAndLabels.map(lambda row: (float(row['prediction']), float(row['class'])))
    metrics = MulticlassMetrics(predictionAndLabels)

    confusion_matrix = metrics.confusionMatrix().toArray()
    tn = confusion_matrix[0][0]
    fn = confusion_matrix[1][0]
    fp = confusion_matrix[0][1]
    tp = confusion_matrix[1][1]

    accuracy = metrics.accuracy
    recall = tp/(tp+fn)
    specificity = tn/(tn+fn)
    fscore = (2*tp)/(2*tp+fp+fn)

    print("Accuracy -> " + str(accuracy))
    print("Recall -> " + str(recall))
    print("Specificity -> " + str(specificity))
    print("F-Score -> " + str(fscore))
    print("AUC -> " + str(auc))
    print("Confusion Matrix -> ")
    print(confusion_matrix)
    print('\n')


if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')

    path_dataset = './filteredC.small.training'
    columns = ['PSSM_central_1_Y', 'PSSM_r2_4_T', 'PSSM_r1_-3_T', 'PSSM_r1_2_T', 'PSSM_r1_0_I', 'PredSS_central']
    balance_type = 'rus'

    # Read Dataset
    sqlc = SQLContext(sc)
    df = sqlc.read.csv(path_dataset, header=True, sep=",", inferSchema=True)

    df_train, df_test = preprocess_dataset(df, columns, balance_type)

    # --------------------- LOGISTIC REGRESSION --------------------- #

    print('Training logistic regresion models...')

    lr_model_1, lr_params_1 = train_logistic_regression(df_train, maxIter=10, regParam=0.1)
    lr_model_2, lr_params_2 = train_logistic_regression(df_train, maxIter=15, regParam=0.1)
    lr_model_3, lr_params_3 = train_logistic_regression(df_train, maxIter=20, regParam=0.01)

    evaluate(lr_model_1, df_test, 'Logistic Regression', lr_params_1)
    evaluate(lr_model_2, df_test, 'Logistic Regression', lr_params_2)
    evaluate(lr_model_3, df_test, 'Logistic Regression', lr_params_3)

    # --------------------- DECISION TREE CLASSIFIER --------------------- #

    print('Training decision tree models...')

    dt_model_1, dt_params_1 = train_random_forest(df_train, maxDepth = 10, impurity = "gini")
    dt_model_2, dt_params_2 = train_random_forest(df_train, maxDepth = 10, impurity = "entropy")
    dt_model_3, dt_params_3 = train_random_forest(df_train, maxDepth = 15, impurity = "gini")

    evaluate(dt_model_1, df_test, 'Decision Tree Classifier', rf_params_1)
    evaluate(dt_model_2, df_test, 'Decision Tree Classifier', rf_params_2)
    evaluate(dt_model_3, df_test, 'Decision Tree Classifier', rf_params_3)

    # --------------------- RANDOM FOREST CLASSIFIER --------------------- #

    print('Training random forest models...')

    rf_model_1, rf_params_1 = train_random_forest(df_train, numTrees = 10, maxDepth = 10, impurity = "gini")
    rf_model_2, rf_params_2 = train_random_forest(df_train, numTrees = 20, maxDepth = 10, impurity = "gini")
    rf_model_3, rf_params_3 = train_random_forest(df_train, numTrees = 20, maxDepth = 15, impurity = "gini")

    evaluate(rf_model_1, df_test, 'Random Forest Classifier', rf_params_1)
    evaluate(rf_model_2, df_test, 'Random Forest Classifier', rf_params_2)
    evaluate(rf_model_3, df_test, 'Random Forest Classifier', rf_params_3)

    sc.stop()

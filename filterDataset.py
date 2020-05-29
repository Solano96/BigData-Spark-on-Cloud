from pyspark import SparkContext, SparkConf, SQLContext


def filter_columns(sc, path_header, path_dataset, columns):
    """ Read data through path header and path data provided"""

    # Get header names
    headers = sc.textFile(path_header).collect()
    inputs = [head for head in headers if "@inputs" in head]
    columns = map(str.strip, columns[0].replace('@inputs', '').split(','))
    columns.append('class')

    # Get data
    sqlc = SQLContext(sc)
    dataset = sqlc.read.csv(path_dataset, header=False, inferSchema=True)

    # Assign column names
    for i in range(0, len(dataset.columns)):
        dataset = dataset.withColumnRenamed(dataset.columns[i], columns[i])

    dataset = dataset.select(columns)
    dataset.write.csv('./filteredC.small.training', header=True, mode="overwrite")


if __name__ == "__main__":
    # Create Spark Context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - Francisco Solano Lopez Rodriguez")
    sc = SparkContext(conf=conf)

    path_header = '/user/datasets/ecbdl14/ECBDL14_IR2.header'
    path_dataset = '/user/datasets/ecbdl14/ECBDL14_IR2.data'
    columns = ['PSSM_central_1_Y', 'PSSM_r2_4_T', 'PSSM_r1_-3_T', 'PSSM_r1_2_T', 'PSSM_r1_0_I', 'PredSS_central', 'class']

    dataset = filter_columns(sc, path_header, path_dataset)

    sc.stop()

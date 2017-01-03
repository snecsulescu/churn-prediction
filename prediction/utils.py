from __future__ import print_function

import logging
import optparse
import sys

from pyspark.context import SparkContext
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.sql.types import *
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def labelData(data, cls):
    return data.map(lambda row: LabeledPoint(cls, [row]))


def load_data_parquet(dir, sqlContext):
    """ Load customers features extracted from orders and websessions
    :return: DataFrame with all the features joint on customerId2
    """
    logger.info("Loading features about websessions from {file}".format(file=get_parket_sessions_features(dir)))
    df_customers_websessions = sqlContext.read.parquet(get_parket_sessions_features(dir))
    df_pd_customers_features = pd.read_pickle(get_pickle_orders_features(dir))
    df_customers_features = sqlContext.createDataFrame(df_pd_customers_features)

    #df_customers_features = sqlContext.read.parquet(get_parket_orders_features(dir))

    df_customers = df_customers_features.join(df_customers_websessions, 'customerId2')
    logger.info(df_customers.dtypes)
    df_customers.dropna()

    return df_customers


def vectorize_data(data):
    """ Prepare the data for the ML system
    :param data: the input DataFrame
    :return: DataFrame with two columns: Label and Features
    """
    return data.map(lambda row: [float(row[1] == 2), Vectors.dense(row[2:])]).toDF(['label', 'features'])


def print_metrics(predictions_and_labels):
    """
    Print the metrics to evaluate a ML system
    :param predictions_and_labels: RDD containing predictions and labels
    :return:
    """
    metrics = MulticlassMetrics(predictions_and_labels)
    print('Confusion Matrix\n', metrics.confusionMatrix().toArray())
    print('Precision of True ', metrics.precision(1))
    print('Precision of False', metrics.precision(0))
    print('Recall of True    ', metrics.recall(1))
    print('Recall of False   ', metrics.recall(0))
    print('F-1 Score         ', metrics.fMeasure())


def getPredictionsLabels(model, test_data):
    """ The predictions for the test_data given a trained model
    :param model: the trained model
    :param test_data: the data to test the system
    :return:
    """
    test_data.map(lambda p: p.features).foreach(print)
    predictions = test_data.map(lambda p: (p.label, model.predict(p.features)))

    return predictions


def subsampling(data):
    """ Subsample a dataset in order to have the ratio positive/negative instances 1:1
    :param data: dataset to subsample
    :return: a subsample of the data dataset
    """
    counts = data.groupBy("label").count()

    counts_0 = counts.where(col('label') == 0).select("count").head()[0]
    counts_1 = counts.where(col('label') == 1).select("count").head()[0]

    ratio = float(counts_1) / counts_0

    data_sampled = data.sampleBy('label', fractions={0: ratio, 1: 1.0})

    return data_sampled


def get_dir_customers(dir):
    return '{dir}/customer/'.format(dir=dir)


def get_dir_receipts(dir):
    return '{dir}/receipts/'.format(dir=dir)


def get_dir_returns(dir):
    return '{dir}/returns/'.format(dir=dir)


def get_dir_websessions(dir):
    return '{dir}/websessions/'.format(dir=dir)


def get_file_orders_features(dir):
    return '{dir}/customers_orders_features.csv'.format(dir=dir)


def get_parket_orders_features(dir):
    return '{dir}/customers_orders_features.parquet'.format(dir=dir)


def get_pickle_orders_features(dir):
    return '{dir}/customers_orders_features.pickle'.format(dir=dir)


def get_file_sessions_features(dir):
    return '{dir}/customers_sessions_features.csv'.format(dir=dir)


def get_parket_sessions_features(dir):
    return '{dir}/customers_sessions_features.parquet'.format(dir=dir)


def arg_dir_subsmpl():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dir',
                      dest="dir",
                      type="string",
                      help="The data directory."
                      )
    parser.add_option('-s', '--subsample',
                      action="store_true",
                      help="Set the option is you want to subsample the training data."
                      )
    options, remainder = parser.parse_args()

    if options.dir is None:
        sys.exit("You must enter the main directory path where the customers, receipts and returns can be found.")

    return options.dir, options.subsample


def arg_dir():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dir',
                      dest="dir",
                      type="string",
                      help="The data directory."
                      )

    options, remainder = parser.parse_args()

    if options.dir is None:
        sys.exit("You must enter the main directory path where the customers, receipts and returns can be found.")

    return options.dir

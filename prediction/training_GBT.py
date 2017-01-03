import logging

from pyspark.context import SparkContext
from pyspark.sql import SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer

from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sc = SparkContext()
sqlContext = SQLContext(sc)

if __name__=="__main__":
    dir, subsmpl = arg_dir_subsmpl()
    data = load_data_parquet(dir,sqlContext)
    data = vectorize_data(data)
    print("#Instances in data:\n")
    data.groupBy("label").count().show()

    (training_data, test_data) = data.randomSplit([0.7, 0.3])
    if subsmpl:
        training_data = subsampling(training_data)

    print("#Instances in trainingData:\n")
    training_data.groupBy("label").count().show()

    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data)

    # train
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, gbt])
    model = pipeline.fit(training_data)

    # test
    predictions = model.transform(test_data)
    predictions.select("prediction", "indexedLabel", "features").show(5)
    predictions.groupBy("prediction", "indexedLabel").count().show()
    gbt_model = model.stages[2]
    print(gbt_model)  # summary only

    # evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    predictions_and_labels = predictions.select("prediction", "label")
    print_metrics(predictions_and_labels.rdd)

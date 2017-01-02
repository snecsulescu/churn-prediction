import logging
import optparse
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
import sys
from utils import load_data, print_metrics, vectorize_data, subsampling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

subsmpl = True
if __name__ == "__main__":

    data = load_data()
    data = vectorize_data(data)
    print("#Instances in data:\n")
    data.groupBy("label").count().show()

    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    print("#Instances in trainingData:\n")
    data.groupBy("label").count().show()

    if subsmpl:
        trainingData = subsampling(trainingData)

    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data)

    # train
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])
    model = pipeline.fit(trainingData)

    # test
    predictions = model.transform(testData)
    predictions.select("prediction", "indexedLabel", "features").show(5)
    predictions.groupBy("prediction", "indexedLabel").count().show()
    dt_model = model.stages[2]
    print(dt_model)  # summary only

    # evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                  metricName="precision")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    predictions_and_labels = predictions.select("prediction", "label")
    print_metrics(predictions_and_labels.rdd)

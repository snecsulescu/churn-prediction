from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

from utils import *

from pyspark.context import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)

if __name__ == "__main__":
    dir, subsmpl = arg_dir_subsmpl()

    data = load_data_parquet(dir, sqlContext)
    data = vectorize_data(data)

    print("#Instances in data:\n")
    data.groupBy("label").count().show()

    (training_data, test_data) = data.randomSplit([0.7, 0.3])

    if subsmpl:
        training_data = subsampling(training_data)

    print("#Instances in trainingData:\n")
    training_data.groupBy("label").count().show()

    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    #scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True)
    lr = LogisticRegression(labelCol="indexedLabel", featuresCol="features", maxIter=20, regParam=0.5)
    pipeline = Pipeline(stages=[label_indexer, lr])
    model = pipeline.fit(training_data)

    for stage in model.stages:
        if isinstance(stage, LogisticRegressionModel):
            # Print the coefficients and intercept for logistic regression
            print("Coefficients: " + str(stage.coefficients))
            print("Intercept: " + str(stage.intercept))

    predictions = model.transform(test_data)
    predictions.select("prediction", "label", "features").show(5)
    predictions.groupBy("prediction", "label").count().show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="precision")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    predictions_and_labels = predictions.select("prediction", "label")

    print_metrics(predictions_and_labels.rdd)

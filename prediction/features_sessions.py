from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import min, mean, max, count, when
from pyspark.sql.types import *

from utils import *

sc = SparkContext()
sqlContext = SQLContext(sc)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_web_sessions(dir):
    """ Load the Websessions table in the memory
    :return: DataFrame with the websessions information useful for the ML system
    """
    customSchema = StructType([ \
        StructField("customerId2", IntegerType(), True), \
        StructField("country", StringType(), True), \
        StructField("startTime", StringType(), True), \
        StructField("site", StringType(), True), \
        StructField("pageViewCount", IntegerType(), True), \
        StructField("nonPageViewEventsCount", IntegerType(), True), \
        StructField("userAgent", StringType(), True), \
        StructField("screenResolution", StringType(), True), \
        StructField("browserSize", StringType(), True), \
        StructField("productViewCount", IntegerType(), True), \
        StructField("productViewsDistinctCount", IntegerType(), True), \
        StructField("productsAddedToBagCount", IntegerType(), True), \
        StructField("productsSavedForLaterFromProductPageCount", IntegerType(), True), \
        StructField("productsSavedForLaterFromCategoryPageCount", IntegerType(), True), \
        StructField("productsPurchasedDistinctCount", IntegerType(), True), \
        StructField("productsPurchasedTotalCount", IntegerType(), True)])

    df = sqlContext.read.format('com.databricks.spark.csv') \
        .options(header='false', delimiter='\t', nullValue='\\N') \
        .load(get_dir_websessions(dir), schema=customSchema)

    return df


def load_customers(dir):
    """ Load the Customers table in the memory
    :return: DataFrame with the customer information
    """
    customSchema = StructType([ \
        StructField("customerId2", IntegerType(), True), \
        StructField("churnlabel", IntegerType(), True), \
        StructField("gender", StringType(), True), \
        StructField("shippingCountry", StringType(), True), \
        StructField("dateCreated", StringType(), True), \
        StructField("yearOfBirth", IntegerType(), True), \
        StructField("premier", IntegerType(), True)])

    df = sqlContext.read.format('com.databricks.spark.csv') \
        .options(header='false', delimiter='\t', nullValue='\\N') \
        .load(get_dir_customers(dir) + '/*', schema=customSchema)

    return df

def features_websessions(df_customers, df_websessions):
    """ Extract from the websessions information features representing the user activity on ASOS site
    :return: DataFrame with the websessions information useful for the ML system
    """
    df_websessions = df_customers.join(df_websessions, "customerId2", 'inner')
    res_counts = df_websessions.groupBy('customerId2').count().alias('nb_sessions')

    res_agg = df_websessions.groupBy('customerId2').agg(
        min('pageViewCount').alias('min_pageViewCount'),
        mean('pageViewCount').alias('mean_pageViewCount'),
        max('pageViewCount').alias('max_pageViewCount'),
        (count(when(df_websessions.pageViewCount != 0, True)) / count('customerId2')).alias('p_not0_pageViewCount'),

        min('nonPageViewEventsCount').alias('min_nonPageViewEventsCount'),
        mean('nonPageViewEventsCount').alias('mean_nonPageViewEventsCount'),
        max('nonPageViewEventsCount').alias('max_nonPageViewEventsCount'),
        (count(when(df_websessions.nonPageViewEventsCount != 0, True)) / count('customerId2')).alias(
            'p_not0_nonPageViewEventsCount'),

        min('productViewCount').alias('min_productViewCount'),
        mean('productViewCount').alias('mean_productViewCount'),
        max('productViewCount').alias('max_productViewCount'),
        (count(when(df_websessions.productViewCount != 0, True)) / count('customerId2')).alias('p_not0_productViewCount'),

        min('productViewsDistinctCount').alias('min_productViewsDistinctCount'),
        mean('productViewsDistinctCount').alias('mean_productViewsDistinctCount'),
        max('productViewsDistinctCount').alias('max_productViewsDistinctCount'),
        (count(when(df_websessions.productViewsDistinctCount != 0, True)) / count('customerId2')).alias(
            'p_not0_productViewsDistinctCount'),

        min('productsAddedToBagCount').alias('min_productsAddedToBagCount'),
        mean('productsAddedToBagCount').alias('mean_productsAddedToBagCount'),
        max('productsAddedToBagCount').alias('max_productsAddedToBagCount'),
        (count(when(df_websessions.productsAddedToBagCount != 0, True)) / count('customerId2')).alias(
            'p_not0_productsAddedToBagCount'),

        min('productsSavedForLaterFromProductPageCount').alias('min_productsSavedForLaterFromProductPageCount'),
        mean('productsSavedForLaterFromProductPageCount').alias('mean_productsSavedForLaterFromProductPageCount'),
        max('productsSavedForLaterFromProductPageCount').alias('max_productsSavedForLaterFromProductPageCount'),
        (count(when(df_websessions.productsSavedForLaterFromProductPageCount != 0, True)) / count('customerId2')).alias(
            'p_not0_productsSavedForLaterFromProductPageCount'),

        min('productsSavedForLaterFromCategoryPageCount').alias('min_productsSavedForLaterFromCategoryPageCount'),
        mean('productsSavedForLaterFromCategoryPageCount').alias('mean_productsSavedForLaterFromCategoryPageCount'),
        max('productsSavedForLaterFromCategoryPageCount').alias('max_productsSavedForLaterFromCategoryPageCount'),
        (count(when(df_websessions.productsSavedForLaterFromCategoryPageCount != 0, True)) / count('customerId2')).alias(
            'p_not0_productsSavedForLaterFromCategoryPageCount'),

        min('productsPurchasedDistinctCount').alias('min_productsPurchasedDistinctCount'),
        mean('productsPurchasedDistinctCount').alias('mean_productsPurchasedDistinctCount'),
        max('productsPurchasedDistinctCount').alias('max_productsPurchasedDistinctCount'),
        (count(when(df_websessions.productsPurchasedDistinctCount != 0, True)) / count('customerId2')).alias(
            'p_not0_productsPurchasedDistinctCount'),

        min('productsPurchasedTotalCount').alias('min_productsPurchasedTotalCount'),
        mean('productsPurchasedTotalCount').alias('mean_productsPurchasedTotalCount'),
        max('productsPurchasedTotalCount').alias('max_productsPurchasedTotalCount'),
        (count(when(df_websessions.productsPurchasedTotalCount != 0, True)) / count('customerId2')).alias(
            'p_not0_productsPurchasedTotalCount'),
    )

    res = res_counts.join(res_agg, 'customerId2')
    return res

if __name__=="__main__":
    dir = arg_dir()

    df_customers = load_customers(dir)
    df_websessions = load_web_sessions(dir)

    df = features_websessions(df_customers, df_websessions)
    logger.info("Features from the customers table: {dtypes}"
                .format(dtypes=df.dtypes))

    df.toPandas().to_csv(get_file_sessions_features(dir), sep='\t', header=True, index=False)
    df.write.parquet(get_parket_sessions_features(dir))
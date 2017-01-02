import os.path
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DIVISIONID_VALS = [4, 5, 6, 7]
SOURCEID_VALS = [1, 2, 3, 4]
PREMIER_VALS = [1, 2, 3, 4, 5, 6]

COLS_CUSTOMERS = ['customerId2', 'churnlabel', 'gender', 'shippingCountry', 'dateCreated', 'yearOfBirth', 'premier']
COLS_RECEIPTS = ['customerId2', 'productId', 'divisionId', 'sourceId', 'itemQty', 'signalDate', 'receiptId', 'price']
COLS_RETURNS = ['customerId2', 'productId', 'divisionId', 'sourceId', 'itemQty', 'signalDate', 'receiptId', 'returnId',
                'returnAction', 'returnReason']

def load_data(inputfile, list_names, parse_dates_list):
    """
    Loads the data necessary for train/test from a csv file into a DataFrame object
    :param inputfile: the input file
    :return: the DataFrame object
    """

    logger.info("loading data...")
    try:
        os.path.exists(inputfile)
    except:
        raise IOError("Input file is missing")

    dataframe = pd.read_csv(inputfile,
                            header=None,
                            names=list_names,
                            parse_dates=parse_dates_list,
                            date_parser=pd.to_datetime,
                            delimiter='\t'
                            )
    logger.info("data loaded")

    return dataframe


def add_column(df_main, serie, name):
    """
    Add the serie Serie as a column in the df_main DataFrame. The join is made based on the customerId2 column
    :param df_main: the DataFrame where the column is added
    :param serie: the new column
    :param name: the name of the new column
    :return: the modified DataFrame
    """
    df = serie.to_frame(name=name)
    df_main_new = df_main.merge(df, left_on='customerId2', right_index='customerId2', how='left')
    return df_main_new


def read_dirs(dir, cols, parse_dates_list):
    """
    Read the
    :param dir:
    :param cols:
    :param parse_dates_list:
    :return:
    """
    list_df = []
    for f in listdir(dir):
        if f.startswith('0'):
            f_path = join(dir, f)
            if isfile(f_path):
                df = load_data(f_path,
                               cols,
                               parse_dates_list,
                               )
                logger.info("Read from {file}.\nShape is {shape}".format(file=f_path, shape=df.shape))
                df.dropna(axis=1)

                list_df.append(df)

    return pd.concat(list_df)


def features_customers(df_customers):
    """ Features that describe the customer
    :param df_customers: DataFrame with customers information
    :return: DataFrame with the customers information useful for the ML system
    """
    for i in PREMIER_VALS:
        k = 'premier_' + str(i)
        df_customers[k] = np.where(df_customers['premier'] == i, 1, 0)

    df_customers['age'] = datetime.now().date().year - df_customers['yearOfBirth']
    df_customers['male'] = np.where(df_customers['gender'] == 'M', 1, 0)
    df_customers['female'] = np.where(df_customers['gender'] == 'F', 1, 0)
    df_customers['days_in_asos'] = (datetime.now().date() - df_customers['dateCreated']).dt.days

    logger.info("Features from the customers table: {shape} {dtypes}"
                .format(shape=df_customers.shape, dtypes=df_customers.dtypes))
    return df_customers


def features_orders(df_customers, df_receipts):
    """ Features that describe the customer orders

    :param df_customers: DataFrame with customers information
    :param df_receipts: DataFrame with customers orders
    :return: DataFrame with the orders information useful for the ML system
    """
    df_customers.sort_values(by=['customerId2'], ascending=[True], inplace=True)
    # total amount of all the orders of a cusrtomer
    df_customers = add_column(df_customers, df_receipts.groupby('customerId2')['price'].sum(), 'total_orders')
    # the min amount paid in one receipt by a customer
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)[
                                  'price'].sum().groupby('customerId2').min()['price'], 'min_order')
    # the mean amount paid per receipt by a customer
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)['price'].sum()
                              .groupby('customerId2').mean()['price'], 'mean_order')
    # the max amount paid per receipt by a customer
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)['price'].sum()
                              .groupby('customerId2').max()['price'], 'max_order')
    # the number of orders
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)['price'].sum()
                              .groupby('customerId2').count()['price'], 'count_orders')

    # the total amount of items bought by a user
    df_customers = add_column(df_customers,
                              df_receipts.groupby('customerId2')['itemQty'].sum(), 'sum_itemQty')
    # the min amount of items bought by a user in a receipt
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)['itemQty'].sum()
                              .groupby('customerId2').min()['itemQty'], 'min_itemQty')
    # the mean amount of items bought by a user in a receipt
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)['itemQty'].sum()
                              .groupby('customerId2').mean()['itemQty'], 'mean_itemQty')
    # the max amount of items bought by a user in a receipt
    df_customers = add_column(df_customers,
                              df_receipts.groupby(['customerId2', 'receiptId'], as_index=False)['itemQty'].sum()
                              .groupby('customerId2').max()['itemQty'], 'max_itemQty')
    # from which dividion type a user buys
    for i in DIVISIONID_VALS:
        k = 'divisionId_' + str(i)
        df_receipts[k] = np.where(df_receipts['divisionId'] == i, 1, 0)
        df_customers = add_column(df_customers, df_receipts.groupby('customerId2')[k].sum(), k)
    # which source type a user uses to pay
    for i in SOURCEID_VALS:
        k = 'sourceId_' + str(i)
        df_receipts[k] = np.where(df_receipts['sourceId'] == i, 1, 0)
        df_customers = add_column(df_customers, df_receipts.groupby('customerId2')[k].sum(), k)

    logger.info("Features from the returns table: {shape} {dtypes}"
                .format(shape=df_customers.shape, dtypes=df_customers.dtypes))
    return df_customers


def features_returns(df_customers, df_receipts, df_returns):
    """ Features representing the orders returned by each client
    :param df_customers: DataFrame with customers information
    :param df_receipts: DataFrame with customers orders
    :param df_returns: DataFrame with customers returns
    :return: DataFrame with the returns information useful for the ML system
    """
    df_customers_returns = pd.merge(df_customers, df_returns, on='customerId2', how='left')

    df_returns['sourceId_10'] = np.where(df_returns['sourceId'] == 10, 1, 0)
    df_customers = add_column(df_customers, df_returns.groupby('customerId2')['sourceId_10'].sum(), 'sourceId_10')
    df_customers.sourceId_10.fillna(0, inplace=True)

    # the returns count for each customer
    has_returns = df_customers_returns[['customerId2', 'returnId']].groupby('customerId2')['returnId'].sum()
    df_customers = add_column(df_customers, has_returns, 'sum_returns')

    # the ratio of items_bought/items_returned
    df_returns_items = df_customers[['customerId2', 'sum_returns', 'sum_itemQty']]
    df_returns_items.set_index('customerId2', inplace=True)
    ratio_items_returns = df_returns_items['sum_itemQty'] / df_returns_items['sum_items_returns']
    ratio_items_returns = np.where(np.isinf(ratio_items_returns), 0, ratio_items_returns)
    df_customers['ratio_items_returns'] = ratio_items_returns

    df_customers.sum_returns.fillna(0, inplace=True)
    df_customers.ratio_items_returns.fillna(0, inplace=True)

    df_customers_receipts = pd.merge(df_customers, df_receipts, on='customerId2')
    df_customers_receipts_returns = pd.merge(df_customers_receipts, df_returns, how='left',
                                             on=['customerId2', 'receiptId', 'productId'])

    # the returns count for each receipt
    df_ids = df_customers_receipts_returns[['customerId2', 'receiptId', 'returnId']]
    receipts_with_returns = df_ids[df_ids.returnId.notnull()] \
        .groupby(['customerId2', 'receiptId']).count().reset_index() \
        .groupby('customerId2').count()['receiptId']
    df_customers = add_column(df_customers, receipts_with_returns, 'receipts_with_returns')
    df_customers.receipts_with_returns.fillna(0, inplace=True)

    # the ratio of count_orders/count_returned
    df_returns_orders = df_customers[['customerId2', 'sum_orders', 'receipts_with_returns']]
    df_returns_orders.set_index('customerId2', inplace=True)
    ratio_orders_with_returns = df_returns_orders['sum_orders'] / df_returns_orders['receipts_with_returns']
    ratio_orders_with_returns = np.where(np.isinf(ratio_orders_with_returns), 0, ratio_orders_with_returns)
    df_customers['ratio_orders_returns'] = ratio_orders_with_returns

    logger.info("Features from the returns table: {shape} {dtypes}"
                .format(shape=df_customers.shape, dtypes=df_customers.dtypes))
    return df_customers


if __name__ == "__main__":

    dir = dir_arg()
    # Load the tables in the memory
    df_customers = read_dirs(get_dir_customers(dir), COLS_CUSTOMERS, [4])
    df_receipts = read_dirs(get_dir_receipts(), COLS_RECEIPTS, [5])
    df_returns = read_dirs(get_dir_returns(), COLS_RETURNS, [5])

    # Extract features from the information provided about customers, their orders and returns
    df_customers = features_customers(df_customers)
    df_customers = features_orders(df_customers, df_receipts)
    df_customers = features_returns(df_customers, df_receipts, df_returns)

    # Drop columns that do not provide useful information for the ML system
    df_customers.drop(['gender', 'shippingCountry', 'dateCreated', 'premier', 'yearOfBirth'], 1, inplace=True)

    logger.info("Final features from the customers, receipts and returns tables: {shape} {dtypes}"
                .format(shape=df_customers.shape, dtypes=df_customers.dtypes))

    # Save the final DataFrame
    df_customers.to_csv(get_file_orders_features(dir), sep='\t', header=True, index=False)
    df_customers.to_pickle(get_pickle_orders_features(dir))
    sc = SparkContext()
    sqlCtx = SQLContext(sc)
    df_spark = sqlCtx.createDataFrame(df_customers)
    df_spark.write.parquet(get_parket_orders_features(dir))

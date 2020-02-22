# -*- coding: utf-8 -*-
#
# Pandas tips and tricks 
# Version 1.0
# Author : Manikandan Jeeva
#
#
# First edited     : 02/22/2020
# Last edited      :
#
# Description      : Code base to support the blog - Pandas tips and tricks
#
# Required input file details :
# 1. Bengaluru_House_Data.csv - found as part of the google datasets (https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data/data)
#
# Output from the code;
# 1. Console output

# TODO: YTS; __Placeholder__

import json;
import datetime;
import csv;
import logging;
import os;
import datetime;
import glob;
import math;
import subprocess;

import numpy as np;
import pandas as pd;
import timeit;

from time import time, sleep;

def delete_old_log_files(delete_flag=False, logger=None, extension_list=None):
    ''' Function to delete the old log files; cleanup process ''';

    directory = './';
    file_list = os.listdir(directory);

    if delete_flag :
        logger.info('DELETE_FLAG is set to true');
        logger.info('All previous logfiles will be deleted');

        logger.info(f'');
        logger.info(f'{"-"*20} File deletion starts here {"-"*20}');
        logger.info(f'');

        fileName = ".".join(__file__.split(".")[:-1]);

        for item in file_list:
            ext_flag = [ item.endswith(i) for i in extension_list ];
            # logger.info(f'{ext_flag} | {item} | {np.sum(ext_flag)} | {fileName in item}');
            if np.sum(ext_flag) and (fileName in item) and (LOG_TS not in item):
                    os.remove(os.path.join(directory, item));
                    logger.info(f'Deleted file : {item}');

        logger.info(f'');
        logger.info(f'{"-"*20} File deletion ends here {"-"*20}');
        logger.info(f'');

    return None


def main(logger=None):
    ''' Main routine to call the entire process flow ''';

    # __Placeholder__ --- Process starts

    logger.info(f'');
    logger.info(f'{"-"*20} __Placeholder__ starts here {"-"*20}');
    logger.info(f'');

    logger.info(f'');
    logger.info(f'{"-"*20} __Placeholder__ ends here {"-"*20}');
    logger.info(f'');

    # __Placeholder__ --- Process ends

    return None;



if __name__ == "__main__":

    LOG_TS = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S');
    LOG_LEVEL = logging.DEBUG;
    # (LogLevel : Numeric_value) : (CRITICAL : 50) (ERROR : 40) (WARNING : 30) (INFO : 20) (DEBUG : 10) (NOTSET : 0)

    DELETE_FLAG = True;
    extension_list = ['.log','.pkl'];  # File extensions to delete after the run;
    ts = time();

    logger = logging.getLogger(__name__);
    logger.setLevel(LOG_LEVEL);
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%d/%m %H:%M:%S');
    fh = logging.FileHandler(filename=f'{".".join(__file__.split(".")[:-1])}_{LOG_TS}.log');
    fh.setLevel(LOG_LEVEL);
    fh.setFormatter(formatter);
    logger.addHandler(fh);

    Test_case = f'Python starter code template with logger and mysql connection : {LOG_TS}';
    Test_comment = '-' * len(Test_case);
    logger.info(pd.__version__)

    logger.info(Test_comment);
    logger.info(Test_case);
    logger.info(Test_comment);

    delete_old_log_files(delete_flag=DELETE_FLAG, logger=logger, extension_list=extension_list);
    # main(logger=logger);
 
    # TODO : Never ever loop through a Pandas data structure

    # Load in ipython console as %%timeit works in the ipython console
    # import pandas as pd

    # def carpet_area_f1(df):
    #     carpet_area = []
    #     for i in range(0, len(df)):
    #         # iterating without using the builtin pandas iterator
    #         c_area = 0.80 * df.iloc[i]['total_sqft']
    #         carpet_area.append(c_area)
    #     return carpet_area

    # def carpet_area_f2(df):
    #     carpet_area = []
    #     for index, row in df.iterrows():
    #         # iterating with builtin pandas iterator
    #         c_area = 0.80 * row['total_sqft']
    #         carpet_area.append(c_area)
    #     return carpet_area

    # %%timeit
    # bengaluru_housing_pd['carpet_area'] = carpet_area_f1(bengaluru_housing_pd)

    # %%timeit
    # bengaluru_housing_pd['carpet_area'] = carpet_area_f2(bengaluru_housing_pd)

    # %%timeit
    # bengaluru_housing_pd['carpet_area'] = bengaluru_housing_pd.apply(lambda row: 0.80 * float(row['total_sqft']), axis=1)

    # TODO : Reduce dataframe memory footprint

    input_file = './Bengaluru_House_Data.csv'
    bengaluru_housing_pd = pd.read_csv(input_file)

    bengaluru_housing_pd.info(memory_usage='deep')

    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 13306 entries, 0 to 13305
    # Data columns (total 9 columns):
    # dtypes: float64(4), object(5)
    # memory usage: 4.5 MB

    required_columns = ['area_type', 'location', 'size', 'total_sqft', 'bath', 'price']
    required_data = pd.read_csv(input_file, usecols=required_columns)
    required_data.info(memory_usage='deep')

    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 13306 entries, 0 to 13305
    # Data columns (total 6 columns):
    # dtypes: float64(3), object(3)
    # memory usage: 2.9 MB

    required_dtype = {'area_type' : 'category'}
    required_data_dtype = pd.read_csv(input_file, usecols=required_columns, dtype=required_dtype)
    required_data_dtype.info(memory_usage='deep')

    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 13306 entries, 0 to 13305
    # Data columns (total 6 columns):
    # dtypes: category(1), float64(3), object(2)
    # memory usage: 2.0 MB

    # TODO : build a dataframe from multiple files row-wise

    from glob import glob

    input_files = sorted(glob('./Bengaluru_House_Data_P*.csv'))
    print(input_files)
    # ['./Bengaluru_House_Data_P1.csv', './Bengaluru_House_Data_P2.csv']

    all_part_pd = pd.concat((pd.read_csv(file) for file in input_files), ignore_index=True)
    all_part_pd.info(memory_usage='deep')
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 26612 entries, 0 to 26611
    # Data columns (total 9 columns):
    # dtypes: float64(4), object(5)
    # memory usage: 9.1 MB

    # TODO : split a df into two random subsets

    train = bengaluru_housing_pd.sample(frac=0.8, random_state=1234) 
    # maintain the seed value to get the same result everytime
    test = bengaluru_housing_pd.drop(train.index)

    print(bengaluru_housing_pd.shape)
    # (13306, 9)
    print(train.shape)
    # (10645, 9)
    print(test.shape)
    # (2661, 9)

    # TODO : Split a string into multiple columns

    bengaluru_housing_pd.rename(columns={'size' : 'property_size'}, inplace=True)

    bengaluru_housing_pd['num_of_bedrooms'] = bengaluru_housing_pd.property_size.str.split(' ', expand=True)[0]
    print(bengaluru_housing_pd.info())
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 13306 entries, 0 to 13305
    # Data columns (total 10 columns):
    # num_of_bedrooms    13290 non-null object
    # dtypes: float64(4), object(6)
    # memory usage: 1.0+ MB
    print(bengaluru_housing_pd.head())
    #               area_type   availability                  location property_size  society  total_sqft  bath  balcony   price num_of_bedrooms
    # 0  Super built-up  Area         19-Dec  Electronic City Phase II         2 BHK  Coomee       1056.0   2.0      1.0   39.07               2
    # 1            Plot  Area  Ready To Move          Chikka Tirupathi     4 Bedroom  Theanmp      2600.0   5.0      3.0  120.00               4
    # 2        Built-up  Area  Ready To Move               Uttarahalli         3 BHK      NaN      1440.0   2.0      3.0   62.00               3
    # 3  Super built-up  Area  Ready To Move        Lingadheeranahalli         3 BHK  Soiewre      1521.0   3.0      1.0   95.00               3
    # 4  Super built-up  Area  Ready To Move                  Kothanur         2 BHK      NaN      1200.0   2.0      1.0   51.00               2

    # TODO : pandas strings to numeric

    bengaluru_housing_pd['num_of_bedrooms'] = pd.to_numeric(bengaluru_housing_pd.num_of_bedrooms, errors='coerce').fillna(0)
    print(bengaluru_housing_pd.info())
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 13306 entries, 0 to 13305
    # Data columns (total 10 columns):
    # num_of_bedrooms    13306 non-null float64
    # dtypes: float64(5), object(5)
    # memory usage: 1.0+ MB

    # TODO : aggregate by multiple functions
    agg_pd = bengaluru_housing_pd.groupby('location').price.agg(['mean', 'std', 'count'])
    print(agg_pd.head(10))

    #                             mean        std  count
    # location                                          
    #  Anekal                16.000000        NaN      1
    #  Banaswadi             35.000000        NaN      1
    #  Basavangudi           50.000000        NaN      1
    #  Bhoganhalli           22.890000        NaN      1
    #  Devarabeesana Halli  124.833333  42.663411      6
    #  Devarachikkanahalli   62.714286  32.281046     14
    #  Electronic City       23.250000   5.303301      2
    #  Mysore Highway        36.875000  23.275076      4
    #  Rachenahalli          23.900000   5.798276      2
    #  Sector 1 HSR Layout  276.000000        NaN      1

    # TODO : convert continuous data into categorical data

    bengaluru_housing_pd['b_categories']= pd.cut(bengaluru_housing_pd.num_of_bedrooms, bins=[0,1,2,4,10,20], labels=['A', 'B', 'C', 'D', 'E'])
    print(bengaluru_housing_pd[['num_of_bedrooms', 'b_categories']].head())
    #    num_of_bedrooms b_categories
    # 0              2.0            B
    # 1              4.0            C
    # 2              3.0            C
    # 3              3.0            C
    # 4              2.0            B

    # TODO : create a pivot table

    pivot_pd = bengaluru_housing_pd.pivot_table(index='b_categories', columns='area_type', values='price', aggfunc='mean')
    print(pivot_pd.head())

    # area_type    (Built-up  Area)(Carpet  Area)(Plot  Area)(Super built-up  Area)
    # b_categories                                                                
    # A                  56.191830     43.885833   66.919231             36.170523
    # B                  56.914291     57.667500   92.812115             57.970348
    # C                 143.229015    113.150395  224.930686            127.725628
    # D                 183.500000    223.000000  253.429701            404.185393
    # E                        NaN           NaN  205.000000            325.000000

    # TODO : Using hdf5 tables to store the data in the disk

    from pandas import HDFStore
    # HDF5 (Hierarchical data format) is a format designed to store large numerical arrays of homogenous type
    # Helps to store large pandas dataframes in disk and retrive them back after session ends

    hdf = HDFStore('./Bengaluru_House_Data.h5')
    hdf.put('bengaluru_data_T1', bengaluru_housing_pd, format='table', data_columns=True, compression='zlib')

    print(bengaluru_housing_pd.shape)
    # (13306, 11)
    print(hdf['bengaluru_data_T1'].shape)
    # (13306, 11)


    logger.info(Test_comment);
    logger.info(f'Code execution took {round((time() - ts), 4)} seconds');
    logger.info(Test_comment);
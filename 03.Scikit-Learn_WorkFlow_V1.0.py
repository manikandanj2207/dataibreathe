# -*- coding: utf-8 -*-
#
# Scikit Learn Machine Learning Process Flow;
# Version 1.0
# Author : <Author names separated by comma; FirstName LastName>
#
# Therapy area     : __Placeholder__
#
# First edited     :
# Last edited      :
#
# Description      : __Placeholder__
#
# Required input file details :
# 1. __Placeholder__
#
# Output from the code;
# 1. __Placeholder__

# TODO: YTS; __Placeholder__

import json;
import datetime;
import csv;
import logging;
import os;
import datetime;
import glob;
import math;

import configparser;
import pymysql;
import subprocess;

import numpy as np;

from time import time, sleep;

def mysql_getConn(db_name=''):
    ''' Will return a mysql connection object to work with database and tables ''';
    config = configparser.ConfigParser()
    config.read("./00.local_settings.ini")

    ms_hostnm = config.get("configuration","hostname")
    ms_usernm = config.get("configuration","username")
    ms_passwd = config.get("configuration","password")

    if db_name == '':
        conn = pymysql.connect(host=ms_hostnm, user=ms_usernm, password=ms_passwd)
    else:
        conn = pymysql.connect(host=ms_hostnm, user=ms_usernm, password=ms_passwd, db=db_name)
    return conn


def mysql_processes(action_to_be_applied):
    ''' Checking whether the mysql process is active or not ''';
    try:
        process_op = subprocess.check_output("mysql.server " + action_to_be_applied, stderr=subprocess.STDOUT, shell=True);
    except Exception as e:
        process_op = str(e);

    return process_op;


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


def main(logger=None):
    ''' Main routine to call the entire process flow ''';

    mysql_status = mysql_processes('status');
    logger.info(f'{mysql_status}');

    mysql_conn = mysql_getConn();
    mysql_cusr = mysql_conn.cursor();

    # __Placeholder__ --- Process starts

    logger.info(f'');
    logger.info(f'{"-"*20} __Placeholder__ starts here {"-"*20}');
    logger.info(f'');


    logger.info(f'');
    logger.info(f'{"-"*20} __Placeholder__ ends here {"-"*20}');
    logger.info(f'');

    # __Placeholder__ --- Process ends

    mysql_conn.close();

    # return None;



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

    logger.info(Test_comment);
    logger.info(Test_case);
    logger.info(Test_comment);

    delete_old_log_files(delete_flag=DELETE_FLAG, logger=logger, extension_list=extension_list);
    main(logger=logger);

    logger.info(Test_comment);
    logger.info(f'Code execution took {round((time() - ts), 4)} seconds');
    logger.info(Test_comment);

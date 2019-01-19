# -*- coding: utf-8 -*-
#
# Kafka producer setup;
# Version 1.0
# Author : Manikandan Jeeva
#
# Therapy area     : NA
#
# First edited     : 12/24/2018
# Last edited      : 12/31/2018
#
# Description      : Kafka Producer setup
#
# Required input file details :
# 1. Require zookeeper and kafka message broker services running
# 2. pip install kafka-python==1.4.4, scikit-learn==0.19.2
#
# Output from the code;
# 1. publishes a clickstream simulation data to kafka stream topic every 5 sec

# Reference:
# 1. Kafka installation quick start : https://kafka.apache.org/quickstart
# 2. Download the latest kafka tgz file and tar it into a folder
# 3. Kafka requires the zookeeper so start the zookeeper processes first
#       cd into unzipped kafka folder;
#       bin/zookeeper-server-start.sh config/zookeeper.properties
# 4. Start the kafka process
#       bin/kafka-server-start.sh config/server.properties

# Reference :
# List of kafka shell commands : https://gist.github.com/ursuad/e5b8542024a15e4db601f34906b30bb5

# TODO: Done: Producer : to generate the clickstream data c1-c10 ten attribute of an ad and a target (0/1)

import json;
import datetime;
import csv;
import logging;
import os;
import datetime;
import glob;
import math;
import requests;
import random;

import subprocess;

import numpy as np;

from time import time, sleep;
from json import dumps;

from kafka import KafkaProducer;

from sklearn import datasets;
from sklearn.preprocessing import MinMaxScaler;


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

    return None;


def check_kafka_prcocess(logger=None):
    ''' Check if the kafka process is running or not ''';
    # All Kafka brokers must be assigned a broker.id. On startup a broker will create an ephemeral node in Zookeeper with a path of /broker/ids/$id ;

    cmd_string = f'echo dump | nc localhost 2181 | grep brokers';
    cmd_output = '';

    try :
        cmd_status = subprocess.check_output(cmd_string, stderr=subprocess.STDOUT, shell=True);
        cmd_output = cmd_status.decode('utf-8').split('\n')[0]
    except Exception as e:
        logger.info(e);

    logger.info(f'Kafka process status : ');

    if len(cmd_output) > 0 :
        logger.info(f'');
        logger.info(f'Running');
        logger.info(f'');
        return 1;
    else:
        logger.info(f'');
        logger.info(f'Not Running');
        logger.info(f'');
        return 0;

    return None;



def list_topics(logger=None, kafka_path=None):
    ''' Run the kafka shell command and return the list of available topics ''';

    # logger.info(f'kafka_path : {kafka_path}');

    cmd_string = f'{kafka_path}bin/kafka-topics.sh --list --zookeeper localhost:2181';
    list_of_topics = subprocess.check_output(cmd_string, stderr=subprocess.STDOUT, shell=True);
    list_of_topics = [i.lower() for i in list_of_topics.decode("utf-8").split("\n") if len(i) > 0 and i.lower() != '__consumer_offsets'];

    logger.info(f'');
    logger.info('List of topics : ');
    logger.info(f'');

    for topic in list_of_topics:
        logger.info(f'{topic}');

    logger.info(f'');

    return list_of_topics;


def delete_all_topics(logger=None, kafka_path=None, list_of_topics=None):
    ''' Run the kafka shell command to delete all listed topics ''';

    logger.info(f'');
    logger.info('Delete all topics : ');
    logger.info(f'');

    for topic in list_of_topics:

        cmd_string = f'{kafka_path}bin/kafka-topics.sh --zookeeper localhost:2181 -delete -topic {topic}';
        cmd_status = subprocess.check_output(cmd_string, stderr=subprocess.STDOUT, shell=True);
        cmd_output = cmd_status.decode('utf-8').split('\n')[0];

        logger.info(f'{cmd_output}');

    return None;


def create_topic(logger=None, kafka_path=None, topic=None):
    ''' Routine will create a new topic; assuming delete_all_topics will run before this routine ''';

    cmd_string = f'{kafka_path}bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic {topic.lower()}';
    cmd_status = subprocess.check_output(cmd_string, stderr=subprocess.STDOUT, shell=True);
    cmd_output = cmd_status.decode('utf-8').split('\n')[0];

    logger.info(f'');
    logger.info(f'{cmd_output}');
    logger.info(f'');

    return None;


def generate_sample(logger=None, number_of_records=None):
    ''' Generate a numpy array with the number of sample row for Xs and Y ''';

    X, y = datasets.make_classification(n_samples=number_of_records, n_features=10, n_informative=6, n_redundant=2, scale=1.0, shift=0.0);
    # X_scaled = MinMaxScaler().fit_transform(X);
    # Fitting a minmaxscalar so that the X values are distributed between 0 and 1;

    return (X, y);


def run_producer(logger=None, topic=None):
    ''' Run a producer to put messages into a topic ''';

    # The function simulates the clickstream data with pass through
    # X0 - X9   : (Dependent) Numerical features which details the attributes of an advertisement
    # y         : (Target) Whether the advertisement resulted in a user click or not
    # for every instance a random choice is made to generate the number of records

    producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'));
    logger.info(f'Publishing messages to the topic : {topic}');

    random_choice_seq = [ i for i in range(100,110) ];
    # MinMaxScaler will not work with 0 or 1 record

    record_count = 0;

    for i in range(10000):
        number_of_records = random.choice(random_choice_seq);
        record_count += number_of_records;
        X, y = generate_sample(logger=logger, number_of_records=number_of_records);

        if i == 0 :
            X_scalar = MinMaxScaler();
            # A single scalar should be used across the entire process sample else creates the scenario for multiple samples
            X_scalar.fit(X);
            X = X_scalar.transform(X);
        else:
            X = X_scalar.transform(X);

        if (i % 500) == 0:
            logger.info(f'Number of messages pushed to message broker : {format(i, "06,")}, record_count : {format(record_count, "09,")}');

        data = {'X' : X.tolist(), 'y' : y.tolist()};
        # Need to convert the ndarray as it is not serializable and it need to be converted tolist();
        producer.send(f'{topic}', value=data);
        sleep(0.05);

    logger.info(f'Closing producer process; Total records generated is {format(record_count, "09,")}');
    return None;



def main(logger=None, kafka_path=None):
    ''' Main routine to call the entire process flow ''';

    # Main call --- Process starts

    logger.info(f'');
    logger.info(f'{"-"*20} List all kafka topics - starts here {"-"*20}');
    logger.info(f'');

    kafka_status = check_kafka_prcocess(logger=logger);

    if kafka_status :

        list_of_topics = list_topics(logger=logger, kafka_path=kafka_path);

        # *************************************** WARNING ***************************************;
        # Do not run the delete_all_topics in a production environment; it will delete all topics;
        # *************************************** WARNING ***************************************;
        delete_all_topics(logger=logger, kafka_path=kafka_path, list_of_topics=list_of_topics);
        # *************************************** WARNING ***************************************;
        # Do not run the delete_all_topics in a production environment; it will delete all topics;
        # *************************************** WARNING ***************************************;

        create_topic(logger=logger, kafka_path=kafka_path, topic='testbroker');
        run_producer(logger=logger, topic='testbroker');


    logger.info(f'');
    logger.info(f'{"-"*20} List all kafka topics - ends here {"-"*20}');
    logger.info(f'');

    # Main call --- Process ends

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

    random.seed(2011);

    kafka_path = '/Users/manikandanjeeva/Documents/Blog_Initiative/Codes/kafka_2.11-2.1.0/';

    Test_case = f'Kafka producer code module : {LOG_TS}';
    Test_comment = '-' * len(Test_case);

    logger.info(Test_comment);
    logger.info(Test_case);
    logger.info(Test_comment);

    delete_old_log_files(delete_flag=DELETE_FLAG, logger=logger, extension_list=extension_list);
    main(logger=logger, kafka_path=kafka_path);


    logger.info(Test_comment);
    logger.info(f'Code execution took {round((time() - ts), 4)} seconds');
    logger.info(Test_comment);

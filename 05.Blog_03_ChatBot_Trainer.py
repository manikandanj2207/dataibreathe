# -*- coding: utf-8 -*-
#
# ChatBot setup and model trainer script
# Version 1.0
# Author : Manikandan Jeeva
#
#
# First edited     : 10/26/2018
# Last edited      :
#
# Description      : Setup part will save the config and domain files;
#                    Trainer part will use the setup files and train the nlu and dialog models
#
# Required input file details :
# 05.Blog_03_ChatBot_domain.yml     : Register the slots, intents and actions along with action templates
# 05.Blog_03_ChatBot_stories.md     : Links the intent with the action setting
# 05.Blog_03_ChatBot_nlu_config.yml : Define the spacy language model en; the pipeline items to be used;
# 05.Blog_03_ChatBot_nlu_data.md    : Define the training data for the intents and also the synonyms for the entities
#                                     Define the data for the registered intents in the domain.yml
# 05.Blog_03_ChatBot_endpoints.yml  : Define the nlu sdk endpoint server details; url and port details
# 05_Blog_03_ChatBot_Actions        : Contains the python script which defines the action details;
#                                     action class name has to be put into domain.yml

# # Works with : rasa-core==0.11.12, rasa-core-sdk==0.11.5, rasa-nlu==0.13.7, sklearn-crfsuite==0.3.6, spacy==2.0.16
#
# Output from the code;
# 1. Trained NLU and Dialogue models;
#
# Server reference: http://fgimian.github.io/blog/2012/12/08/setting-up-a-rock-solid-python-development-web-server/

# TODO: YTS; Train the mode for the bitCoin API

import json;
import datetime;
import csv;
import logging;
import os;
import datetime;
import glob;
import math;
import subprocess;
import warnings;
import sys;

import numpy as np;

from time import time, sleep;
from contextlib import redirect_stdout;

from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

def nlu_config_dump(filename_p1=None, logger=None):
    """ Write the nlu config yml file from the string """;

    temp_str ="""
language: "en"

pipeline:
- name: "nlp_spacy"
- name: "tokenizer_spacy"
- name: "intent_entity_featurizer_regex"
- name: "intent_featurizer_spacy"
- name: "ner_crf"
- name: "ner_spacy"
- name: "ner_synonyms"
- name: "intent_classifier_sklearn"
"""

    with open(f'./{filename_p1}_nlu_config.yml', 'w') as fb:
        fb.write(temp_str.lstrip().rstrip());

    logger.info(f'$ConfigFile$ : {filename_p1}_nlu_config.yml has been written successfully');

    return f'./{filename_p1}_nlu_config.yml';


def nlu_data_dump(filename_p1=None, logger=None):
    """ Write the nlu data md file from the string """;

    temp_str ="""
## intent:greeting
- hey
- hello
- hi
- good morning
- good day
- good afternoon
- morning
- good evening

## intent:do_you_have_friends
- Do you have any friends
- Do u have any friend
- Are you friends with anyone
- Any friends u have
- Do you have a friend
- Do u have a friend
- Are you a good friend

## intent:how_are_you
- how are you
- how is it going
- how do you do
- how r u
- how you
- how are u
- Whats up
- how are you doing

## intent:bye
- bye
- good bye
- bye bye
- good by
- ok good bye
- see ya
- c u
- cu
- see you later
- laters
- good night
- cee you later
- goodbye
- have a nice one
- see you around

## intent:get_weather
- what's the weather
- what's the weather in [Chennai](GPE)
- what is the weather
- what is the weather in [Delhi](GPE)
- what is the weather in [Bengaluru](GPE)
- what is the weather in [Mumbai](GPE)
- what is the weather in [Coimbatore](GPE)
- whats the weather
- what is the weather like
- how is the weather in [Kolkota](GPE)
- how's the weather in [Chennai](GPR)
- hows the weather
- weather in [Hyderabad](GPE)
- weather [Guntor](GPE)

## intent:my_name_is
- I am [Ravi](PERSON)
- I am [Ashok](PERSON)
- I'm [Tom](PERSON)
- im [Rahul](PERSON)
- I am [Ramesh Kumar](PERSON)
- My name is [Pradeep Kumar](PERSON)
- my nam is [Raj Kumar](PERSON)

"""

    with open(f'./{filename_p1}_nlu_data.md', 'w') as fb:
        fb.write(temp_str.lstrip().rstrip());

    logger.info(f'$ConfigFile$ : {filename_p1}_nlu_data.md has been written successfully');

    return f'./{filename_p1}_nlu_data.md';


def domain_dump(filename_p1=None, logger=None):
    """ Write the domain yml file from the string """;

    temp_str ="""
slots:
  PERSON:
    type: text
  GPE:
    type: text

intents:
  - greeting
  - how_are_you
  - bye
  - do_you_have_friends
  - my_name_is
  - get_weather

actions:
  - utter_greeting
  - utter_how_i_am
  - utter_friends
  - utter_bye
  - utter_its_nice_to_meet_you
  - action_get_weather
  #- actions.weather.ActionGetWeather

templates:
  utter_default:
    - I am a ChatBot.
  utter_greeting:
    - Hi!
    - Hello!
    - Hello, my friend.
    - Hello there, my friend.
  utter_bye:
    - Bye.
    - Good bye.
  utter_how_i_am:
    - I am doing good
    - Fine, thanks
  utter_friends:
    - I have a lot of friends
    - Yeah I do have friends
    - Yes I do have a lot of friends
  utter_its_nice_to_meet_you:
    - It's nice to meet you, {PERSON}.
    - Nice to meet you, {PERSON}.

"""

    with open(f'./{filename_p1}_domain.yml', 'w') as fb:
        fb.write(temp_str.lstrip().rstrip());

    logger.info(f'$ConfigFile$ : {filename_p1}_domain.yml has been written successfully');

    return f'./{filename_p1}_domain.yml';


def stories_dump(filename_p1=None, logger=None):
    """ Write the stories md file from the string """;

    temp_str ="""
## greeting
* greeting
- utter_greeting

## my friends
* do_you_have_friends
- utter_friends

## my name is
* my_name_is
- utter_its_nice_to_meet_you

## how are you
* how_are_you
- utter_how_i_am

## good bye
* bye
- utter_bye

## get weather
* get_weather
- action_get_weather

"""
    with open(f'./{filename_p1}_stories.md', 'w') as fb:
        fb.write(temp_str.lstrip().rstrip());

    logger.info(f'$ConfigFile$ : {filename_p1}_stories.md has been written successfully');

    return f'./{filename_p1}_stories.md';


def endpoint_dump(filename_p1=None, logger=None):
    """ Write the stories md file from the string """;

    temp_str ="""
action_endpoint:
  url: "http://localhost:5055/webhook"
"""

    with open(f'./{filename_p1}_endpoints.yml', 'w') as fb:
        fb.write(temp_str.lstrip().rstrip());

    logger.info(f'$ConfigFile$ : {filename_p1}_endpoints.yml has been written successfully');

    return f'./{filename_p1}_endpoints.yml';


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

    # Train and persist NLU model --- Process starts

    logger.info(f'');
    logger.info(f'{"-"*20} Train and persist NLU model starts here {"-"*20}');
    logger.info(f'');

    filename_p1 = "_".join(__file__.split("_")[:-1]);

    nlu_config = nlu_config_dump(filename_p1=filename_p1, logger=logger);
    nlu_data = nlu_data_dump(filename_p1=filename_p1, logger=logger);

    warnings.filterwarnings('ignore');
    warnings.simplefilter(action='ignore', category=FutureWarning);
    # warnings.simplefilter(action='ignore', category=DP);

    training_data = load_data(f'{nlu_data}');
    trainer = Trainer(config.load(f'{nlu_config}'));
    trainer.train(training_data);
    model_directory = trainer.persist(f'{filename_p1}_models/nlu/', fixed_model_name="current");

    logger.info(f'');
    logger.info(f'{"-"*20} Train and persist NLU model ends here {"-"*20}');
    logger.info(f'');

    # Train and persist NLU model --- Process ends

    # Dialogue model --- Process starts

    logger.info(f'');
    logger.info(f'{"-"*20} Dialogue model starts here {"-"*20}');
    logger.info(f'');

    domain_file = domain_dump(filename_p1=filename_p1, logger=logger);
    stories_file = stories_dump(filename_p1=filename_p1, logger=logger);
    endpoint_file = endpoint_dump(filename_p1=filename_p1, logger=logger);

    subprocess.call('export PYTHONWARNINGS="ignore"', shell=True);

    # python3 -m rasa_core_sdk.endpoint --actions 04_Blog_03_ChatBot_Actions.weather
    return_code = subprocess.call("python3 -m rasa_core_sdk.endpoint --actions 05_Blog_03_ChatBot_Actions.weather &", shell=True);
    logger.info(f'{return_code}');

    agent = Agent( domain_file, policies=[MemoizationPolicy(max_history=3), KerasPolicy()] );
    training_data = agent.load_data(stories_file);

    agent.train( training_data, epochs=400, batch_size=100, validation_size=0.2 );
    agent.persist(f'{filename_p1}_models/dialogue/');

    # python3 04.Blog_03_ChatBot_Run.py --core ./models/dialogue --nlu ./models/nlu/default/current --endpoints 04.Blog_03_endpoints.yml
    return_code = subprocess.call(f'python3 05.Blog_03_ChatBot_Run.py --core ./{filename_p1}_models/dialogue --nlu ./{filename_p1}_models/nlu/default/current --endpoints {endpoint_file}', shell=True);

    logger.info(f'');
    logger.info(f'{"-"*20} Dialogue model ends here {"-"*20}');
    logger.info(f'');

    # Dialogue model --- Process ends

    # return None;


if __name__ == "__main__":

    LOG_TS = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S');
    # LOG_LEVEL = logging.DEBUG;
    LOG_LEVEL = logging.ERROR;
    # (LogLevel : Numeric_value) : (CRITICAL : 50) (ERROR : 40) (WARNING : 30) (INFO : 20) (DEBUG : 10) (NOTSET : 0)

    DELETE_FLAG = True;
    extension_list = ['.log','.pkl'];  # File extensions to delete after the run;
    ts = time();

    logger = logging.getLogger(__name__);
    logger.setLevel(LOG_LEVEL);
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%d/%m %H:%M:%S');
    args=(sys.stdout,)
    fh = logging.FileHandler(filename=f'{".".join(__file__.split(".")[:-1])}_{LOG_TS}.log');
    fh.setLevel(LOG_LEVEL);
    fh.setFormatter(formatter);
    logger.addHandler(fh);
    logger.write = lambda msg: logger.info(msg) if msg != '\n' else None;

    Test_case = f'ChatBot setup and model trainer script : {LOG_TS}';
    Test_comment = '-' * len(Test_case);

    logger.info(Test_comment);
    logger.info(Test_case);
    logger.info(Test_comment);

    delete_old_log_files(delete_flag=DELETE_FLAG, logger=logger, extension_list=extension_list);
    main(logger=logger);

    logger.info(Test_comment);
    logger.info(f'Code execution took {round((time() - ts), 4)} seconds');
    logger.info(Test_comment);

# -*- coding: utf-8 -*-
#
# Scikit Learn Machine Learning Process Flow;
# Version 1.0
# Author : Manikandan Jeeva
#
#
# First edited     : 27 September 2018
# Last edited      :
#
# Description      : Scikit Learn Machine Learning Basics Blog WorkFlow
#
# Required input file details :
# 1. __Placeholder__
#
# Output from the code;
# 1. __Placeholder__

# TODO: YTS;

import datetime;
import csv;
import logging;
import os;
import datetime;

import numpy as np;
import pandas as pd;

from time import time, sleep;

from sklearn import model_selection;
from sklearn import metrics;
from sklearn.metrics import make_scorer;
from sklearn import preprocessing;

from sklearn import datasets;

from sklearn.dummy import DummyClassifier;

from sklearn.decomposition import KernelPCA;

from sklearn.feature_selection import SelectKBest, chi2, f_classif;

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV;

from sklearn import linear_model;

from sklearn.svm import SVC;

import matplotlib;
import matplotlib.pyplot as plt;
import matplotlib.style as style;
# Check for style.available for more styles;

import seaborn as sns;

# matplotlib.rcParams['font.family'] = 'sans-serif';
# matplotlib.rcParams['font.sans-serif'] = ['Verdana'];

# matplotlib.rcParams['font.family'] = 'cursive';
#
# matplotlib.rcParams['font.weight'] = 8;
# matplotlib.rcParams['font.size'] = 9.5;

matplotlib.rcParams['font.family'] = 'fantasy';

matplotlib.rcParams['font.weight'] = 3;
matplotlib.rcParams['font.size'] = 10;

# ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’
# https://matplotlib.org/users/text_props.html

style.use('bmh');
# style.use('seaborn-paper');
# style.use('seaborn-deep');

from itertools import chain;

def convert2pandas_df(x_array=None, y=None, feature_names=None, target_name=None):
    ''' list of datasets part of the sklearn ''';

    assert x_array.shape[1] == len(feature_names);  # assert the length of x_array and column label length are same;
    assert x_array.shape[0] == len(y); # The target length should equal the features length;
    assert type(y) == type([]); # Target should of the type list;
    assert type(feature_names) == type([]); # feature_names should of the type list;

    data_dict = {};
    data_dict[target_name] = y;

    for i, col_name in enumerate(feature_names):
        data_dict[col_name] = list(chain.from_iterable( x_array[:, [i]] ));

    return pd.DataFrame(data_dict);


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


def chart_save_image(plt=None, f_size=None, left=None, right=None, bottom=None, top=None, wspace=None, hspace=None, fileName=None):
    ''' Save the chart image with the set of specific options ''';

    fig = plt.gcf()
    fig.set_size_inches(8, 4.5) # To maintain the 16:9 aspect ratio;

    if f_size :
        fig.set_size_inches(f_size[0], f_size[1]);

    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplots_adjust

    # left          = 0.125     # the left side of the subplots of the figure
    # right         = 0.9       # the right side of the subplots of the figure
    # bottom        = 0.125     # the bottom of the subplots of the figure
    # top           = 0.9       # the top of the subplots of the figure
    # wspace        = 0.0       # the amount of width reserved for blank space between subplots,
    #                           # expressed as a fraction of the average axis width
    # hspace        = 0.0       # the amount of height reserved for white space between subplots,
    #                           # expressed as a fraction of the average axis height

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace);
    plt.savefig(f'{fileName}');
    plt.clf();

    # plt.savefig(f'./Make_Moons_Image.png', bbox_inches='tight', pad_inches = 1);

def cost_accuracy(actual, prediction):
    """ Custom accuracy cost function to be used in the scorer """;
    # accuracy = correct predictions / total predictions

    assert len(actual) == len(prediction);

    return round((np.sum(actual == prediction) / len(actual)) , 4);

def main(logger=None):
    ''' Main routine to call the entire process flow ''';

    # Load_Dataset --- Process starts

    logger.info(f'');
    logger.info(f'{"-"*20} Load dataset starts here {"-"*20}');
    logger.info(f'');

    # TODO: DONE; Load Cancer dataset;

    cancer_data_dict = datasets.load_breast_cancer();
    cancer_data_pd = convert2pandas_df(x_array=cancer_data_dict['data'],
                      y=[ cancer_data_dict['target_names'][i] for i in cancer_data_dict['target'] ],
                      # feature_names=iris_dict['feature_names'],
                      feature_names=list(cancer_data_dict['feature_names']),
                      target_name='Target');

    # logger.info(f'{cancer_data_pd.head()}');

    sns.lmplot( x="area error", y="compactness error", data=cancer_data_pd, fit_reg=False, hue='Target', legend=False,
               palette=dict(malignant="#BF0C2B", benign="#02173E")); # , versicolor="#F5900E"));
    plt.legend(loc='lower right');
    chart_save_image(plt=plt, f_size=(8, 8), left=0.125, right=0.9, bottom=0.125, top=0.9, wspace=0.0, hspace=0.0, fileName='./Cancer_Data_Plot.png');

    selected_columns = ['Target', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity',
                        'mean concave points', 'mean symmetry'];

    g = sns.pairplot(cancer_data_pd[selected_columns], hue="Target", diag_kind="kde",  palette=dict(malignant="#BF0C2B", benign="#02173E"), diag_kws=dict(shade=True));
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False);
    chart_save_image(plt=plt, f_size=(16, 16), left=0.05, right=0.97, bottom=0.05, top=0.97, wspace=0.02, hspace=0.02, fileName='./Cancer_Data_PairPlot.png');

    logger.info(f'');
    logger.info(f'{"-"*20}  Load dataset ends here {"-"*20}');
    logger.info(f'');

    # Load_Dataset --- Process ends

    # __Placeholder__ --- Process Starts

    # TODO: DONE; 001; Train test split; stratified;
    X_train, X_test, y_train, y_test = train_test_split(cancer_data_pd[cancer_data_dict.feature_names],
                                                        # cancer_data_pd['Target'],
                                                        cancer_data_dict['target'], # Has to be binary for scorer F1 and Percision;
                                                        test_size=0.20,
                                                        # stratify=cancer_data_pd['Target'],
                                                        stratify=cancer_data_dict['target'],
                                                        random_state=111,
                                                        shuffle=True);

    logger.info(f'{X_train.shape} {type(X_train)} {X_train.columns}');
    logger.info(f'{X_test.shape}');
    logger.info(f'{y_train.shape}');
    logger.info(f'{y_test.shape}');

    # TODO: DONE; 002; Dummy Classifier ;

    # dummy_classifier = DummyClassifier(strategy="stratified");
    dummy_classifier = DummyClassifier(strategy="most_frequent");

    # TODO: DONE; 003; Cross_over_score and predict and Metrics (make_scorer)

    accuracy_scorer = make_scorer(cost_accuracy, greater_is_better=True);

    kfold = model_selection.KFold(n_splits=10, random_state=111);
    # results = model_selection.cross_val_score(dummy_classifier, X_train, y_train, cv=kfold, scoring='accuracy');
    # logger.info(f'{results} {np.mean(results)} {np.var(results)} {np.std(results)}');

    results = model_selection.cross_val_score(dummy_classifier, X_train, y_train, cv=kfold, scoring=accuracy_scorer);
    logger.info(f'{results} {np.mean(results)} {np.var(results)} {np.std(results)}');

    DummyClassifier_mean = np.mean(results);

    # TODO: DONE; 004; Standardization ;

    # std_scaler = preprocessing.StandardScaler();  # Contains the negative values
    std_scaler = preprocessing.MinMaxScaler(); # Range between 0 to 1; No negative terms;
    std_scaler = std_scaler.fit(X_train);
    scaled_X_train = pd.DataFrame(std_scaler.transform(X_train), columns=X_train.columns);

    logger.info(f'{X_train["mean radius"].describe()}');
    logger.info(f'{scaled_X_train["mean radius"].describe()}');

    # TODO: DONE; 005; SelectKBest; Feature selection ;

    # selectKbest_est = SelectKBest(chi2, k=4); f_classif
    selectKbest_est = SelectKBest(f_classif, k=8);
    selectKbest_X_train = selectKbest_est.fit_transform(X_train, y_train);

    logger.info(f'{selectKbest_est.get_params(deep=True)}');
    logger.info(f'{selectKbest_est.get_support(indices=False)}');
    logger.info(f'{selectKbest_est.get_support(indices=True)}');
    logger.info(f'{X_train.columns[selectKbest_est.get_support(indices=True)]}');

    # TODO: DONE; 006; Polynomial Features ;

    poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False, interaction_only=False);
    X_train_poly = poly.fit_transform(X_train);
    X_train_p2 = pd.DataFrame(X_train_poly, columns=poly.get_feature_names(X_train.columns));

    lr = linear_model.LogisticRegression(fit_intercept=False, random_state=111);
    results = model_selection.cross_val_score(lr, X_train_p2, y_train, cv=kfold, scoring=accuracy_scorer); # , verbose=True);

    imp_percentage = round((np.mean(results) - DummyClassifier_mean) / DummyClassifier_mean, 4);

    logger.info(f'DummyClassifier accuracy : {DummyClassifier_mean}');
    logger.info(f'LogisticRegression accuracy : {np.mean(results)}');

    logger.info(f'The improvement over the DummyClassifier is : {imp_percentage}');

    # TODO: DONE; 007; Kernel PCA ;

    # kernel_param = ('rbf', 0.25);
    kernel_param = ('rbf', 1);

    kpca = KernelPCA(n_components=4, kernel=kernel_param[0], gamma=kernel_param[1], fit_inverse_transform=True, random_state=111) # n_jobs=-1,
    kpca.fit(scaled_X_train);   # The data has to be scaled;
    kpca_X_train = kpca.transform(scaled_X_train);

    lr = linear_model.LogisticRegression(fit_intercept=False, random_state=111);
    results = model_selection.cross_val_score(lr, kpca_X_train, y_train, cv=kfold, scoring=accuracy_scorer); # , verbose=True);

    imp_percentage = round((np.mean(results) - DummyClassifier_mean) / DummyClassifier_mean, 4);

    logger.info(f'DummyClassifier accuracy : {DummyClassifier_mean}');
    logger.info(f'LogisticRegression accuracy : {np.mean(results)}');

    logger.info(f'The improvement over the DummyClassifier is : {imp_percentage}');

    # TODO: DONE; 008; Grid-Search ;

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}];

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring=accuracy_scorer);
    clf.fit(X_train, y_train);

    logger.info(f'Best parameters set found on development set: {clf.best_params_}');
    logger.info('');
    logger.info('Grid scores on development set:');
    logger.info('');
    means = clf.cv_results_['mean_test_score'];
    stds = clf.cv_results_['std_test_score'];
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        logger.info(f'{round(mean,2)} (+/-{round(std*2,2)}) for {params}');
    logger.info('');

    logger.info('Detailed classification report:');
    logger.info('');
    logger.info('The model is trained on the full development set.');
    logger.info('The scores are computed on the full evaluation set.');
    logger.info('');
    y_true, y_pred = y_test, clf.predict(X_test);
    logger.info(f'{metrics.classification_report(y_true, y_pred)}');
    logger.info('');

    # TODO: YTS; 009; Customer Transformer for the pipeline ;

    # TODO: YTS; 010; Pipeline ;

    # TODO: YTS; 011; One-hot encoder; Label Encoder; Binary Encoder;

    # TODO: YTS; 012; Feature Extraction from image and text;

    # TODO: YTS; 013; Ensemble and BaseClone;

    # TODO: YTS; 014; Utils for dumping and loading models;



    # __Placeholder__ --- Process ends


if __name__ == "__main__":

    LOG_TS = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S');
    LOG_LEVEL = logging.DEBUG;
    # (LogLevel : Numeric_value) : (CRITICAL : 50) (ERROR : 40) (WARNING : 30) (INFO : 20) (DEBUG : 10) (NOTSET : 0)

    DELETE_FLAG = True;
    extension_list = ['.log'];  # File extensions to delete after the run;
    ts = time();

    logger = logging.getLogger(__name__);
    logger.setLevel(LOG_LEVEL);
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%d/%m %H:%M:%S');
    fh = logging.FileHandler(filename=f'{".".join(__file__.split(".")[:-1])}_{LOG_TS}.log');
    fh.setLevel(LOG_LEVEL);
    fh.setFormatter(formatter);
    logger.addHandler(fh);

    Test_case = f'Scikit Learn Machine Learning Workflow Process Code : {LOG_TS}';
    Test_comment = '-' * len(Test_case);

    logger.info(Test_comment);
    logger.info(Test_case);
    logger.info(Test_comment);

    delete_old_log_files(delete_flag=DELETE_FLAG, logger=logger, extension_list=extension_list);
    main(logger=logger);

    logger.info(Test_comment);
    logger.info(f'Code execution took {round((time() - ts), 4)} seconds');
    logger.info(Test_comment);


# Temp --- Process starts
# Testing and unused code blocks for reference;
# Temp --- Process ends
#
# iris_dict = datasets.load_iris();
# iris_pd = convert2pandas_df(x_array=iris_dict['data'],
#                   y=[ iris_dict['target_names'][i] for i in iris_dict['target'] ],
#                   # feature_names=iris_dict['feature_names'],
#                   feature_names=['petal_length', 'petal_width', 'sepal_length', 'sepal_width'],
#                   target_name='Flower_Class');
#
# logger.info(f'{iris_pd.head()}');
#
# moons_tuple = datasets.make_moons(n_samples=100);
# moons_pd = convert2pandas_df(x_array=moons_tuple[0],
#                   y=list(moons_tuple[1]),
#                   feature_names=['x_cord', 'y_cord'],
#                   target_name='category');
#
# logger.info(f'{moons_pd.head()}');
#
# # sns.lmplot( x="x_cord", y="y_cord", data=moons_pd, fit_reg=False, hue='category', legend=False);
# sns.lmplot( x="petal_length", y="petal_width", data=iris_pd, fit_reg=False, hue='Flower_Class', legend=False,
#            # palette=dict(setosa="#9b59b6", virginica="#3498db", versicolor="#95a5a6"));
#            # palette=dict(setosa="#BF0C2B", virginica="#02173E", versicolor="#F14C13"));
#            palette=dict(setosa="#BF0C2B", virginica="#02173E", versicolor="#F5900E"));
#            # Kopie von Copy of コピー rocket021x; Adobe color palette;
#
# plt.legend(loc='lower right');
#
# chart_save_image(plt=plt, f_size=(8, 8), left=0.125, right=0.9, bottom=0.125, top=0.9, wspace=0.0, hspace=0.0, fileName='./Iris_Plot.png');
#
# tips = sns.load_dataset("tips");
# sns.violinplot(x = "total_bill", data=tips);
# chart_save_image(plt=plt, f_size=None, left=0.125, right=0.9, bottom=0.125, top=0.9, wspace=0.0, hspace=0.0, fileName='./Violin_Plot.png');
#
# blobs_tuple = datasets.make_blobs(n_samples=500);
# blobs_pd = convert2pandas_df(x_array=blobs_tuple[0],
#                   y=list(blobs_tuple[1]),
#                   feature_names=['x_cord', 'y_cord'],
#                   target_name='category');
#
# sns.lmplot( x="x_cord", y="y_cord", data=blobs_pd, fit_reg=False, hue='category', legend=False);
# plt.legend(loc='lower right');
# chart_save_image(plt=plt, f_size=None, left=0.125, right=0.9, bottom=0.125, top=0.9, wspace=0.0, hspace=0.0, fileName='./Blob_Plot.png');
#
# g = sns.pairplot(iris_pd, hue="Flower_Class", diag_kind="kde", palette=dict(setosa="#BF0C2B", virginica="#02173E", versicolor="#F5900E"), diag_kws=dict(shade=True));
# for i, j in zip(*np.triu_indices_from(g.axes, 1)):
#     g.axes[i, j].set_visible(False)
# chart_save_image(plt=plt, f_size=(16, 16), left=0.05, right=0.97, bottom=0.05, top=0.97, wspace=0.05, hspace=0.06, fileName='./Iris_PairPlot.png');
# # Left and Right are in pixel percentage positions;
#
# # JointPlot with HEX Bins
# # with sns.axes_style('white'):
# #     sns.jointplot("x_cord", "y_cord", blobs_pd, kind='hex');
#
# sns.distplot(iris_pd['petal_length']);
# chart_save_image(plt=plt, left=0.05, right=0.97, bottom=0.05, top=0.97, wspace=0.05, hspace=0.06, fileName='./Joint_Plot.png');

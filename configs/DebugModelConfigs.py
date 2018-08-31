#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Jun 28 2018

@author: cttsai
'''

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer


cv_type = 'StratifiedKFold'
n_jobs = -1


#default loc at file_dir_path['data']
fileFeatureImportance = [

    ]

ModelConfigs = {
    'LossGuideXGB': {
        'model': XGBClassifier,

        'hyperparameters'     : {}, # obtained from finetuned
        'hyperparameter_optimization': {
            #skopt itself
            'search_settings':{
                'n_calls'     : 2,
                'n_inits'     : 1,
                'random_state': 42,
            },
            #cv settings
            'evaluation_settings': {
                'validation'  : 'StratifiedKFold',
                'nr_fold'     : 3,    # CV
                'nr_splits'   : 3,    # Splits
                'train_size'  : 0.5,  # split size
                'valid_size'  : 0.5,  # split size
                'split_seed'  : 538,
                'eval_metric' : 'roc_auc',  # sklearn scorer
            },
            #
            'initialize': {
                'objective'        : 'binary:logistic',
                'booster'          : 'gbtree',
#               'n_jobs'           :  n_jobs,
                'n_estimators'     : 200,
                'tree_method'      : 'hist',
                'grow_policy'      : 'lossguide',
                'max_depth'        : 7,  # deafult=6
                'base_score'       : 0.95,
                'max_delta_step'   : 3,  #default=0
            },
            #skopt
            'search_space': {
                'scale_pos_weight' : Real(2, 16, 'log-uniform'),
#                'max_depth'        : Integer(8, 15),
                'learning_rate'    : Real(1e-3, 1e-1, 'log-uniform'),
                'max_leaves'       : Integer(11,  47),
                'min_child_weight' : Integer(2, 64),
                'gamma'            : Real(1e-4, 1e-1, 'log-uniform'),  # default=0
                'subsample'        : Real(0.6, 0.9),
                'colsample_bytree' : Real(0.5, 0.9),
                'reg_alpha'        : Real(1e-5, 1e-2, 'log-uniform'),  # default=0
                'reg_lambda'       : Real(1e-2, 1e1, 'log-uniform'),  # default=1
            },
        },
    },  # LossGuideXGB

    #sample lgbm
    'LGBM': {
        'model': LGBMClassifier,
        'hyperparameter_optimization': {
            #skopt itself
            'search_settings':{
                'n_calls'     : 2,
                'n_inits'     : 1,
                'random_state': 42,
            },
            #cv settings
            'evaluation_settings': {
                'validation'  : cv_type,
                'nr_fold'     : 3,    # CV
                'nr_splits'   : 3,    # Splits
                'train_size'  : 0.5,  # split size
                'valid_size'  : 0.5,  # split size
                'split_seed'  : 538,
                'eval_metric' : 'roc_auc',  # sklearn scorer
            },
            #
            'initialize': {
                'device'            : 'cpu',
                'objective'         : 'binary',
                'boosting_type'     : 'gbdt',
                'n_jobs'            : n_jobs,
                'max_depth'         : 8,
                'n_estimators'      : 1000,
                'subsample_freq'    : 2,
                'subsample_for_bin' : 200000,
                'min_data_per_group': 100,  #default=100
                'max_cat_to_onehot' : 4,    #default=4
                'cat_l2'            : 10.,  #default=10
                'cat_smooth'        : 10.,  #default=10
                'max_cat_threshold' : 32,   #default=32
                'metric_freq'       : 10,
                'verbosity'         : -1,
                'metric'            : 'auc',
#                'metric'            : 'binary_logloss',
            },
            #skopt
            'search_space': {
                'num_leaves'        : Integer(15, 63),
                'learning_rate'     : Real(1e-3, 1e-1, 'log-uniform'),
                'scale_pos_weight'  : Real(2, 16, 'log-uniform'),
                'min_split_gain'    : Real(1e-4, 1e-1, 'log-uniform'),  # defult=0
                'min_child_weight'  : Real(1e-2, 1e2, 'log-uniform'),  # defaul=1e-3
                'min_child_samples' : Integer(10, 80),  # defult=20
                'subsample'         : Real(0.6, 0.9),
                'colsample_bytree'  : Real(0.5, 0.9),
                'reg_alpha'         : Real(1e-5, 1e-1, 'log-uniform'),  # defult=0
                'reg_lambda'        : Real(1e-4, 1e-0, 'log-uniform'),  # defult=0
                'cat_l2'            : Real(1e0, 1e2, 'log-uniform'),  #default=10
                'cat_smooth'        : Real(1e0, 1e2, 'log-uniform'),  #default=10
            },
        },
    },  # LGBM

    'ScikitRF': {
        'model': RandomForestClassifier,
        'hyperparameter_optimization': {
            #skopt itself
            'search_settings':{
                'n_calls'     : 2,
                'n_inits'     : 1,
                'random_state': 42,
            },
            #cv settings
            'evaluation_settings': {
                'validation'  : 'StratifiedKFold',
                'nr_fold'     : 3,    # CV
                'nr_splits'   : 3,    # Splits
                'train_size'  : 0.5,  # split size
                'valid_size'  : 0.5,  # split size
                'split_seed'  : 538,
                'eval_metric' : 'roc_auc',  # sklearn scorer
            },
            #
            'initialize': {
                'criterion'   : 'entropy', #'gini',
                'oob_score'   : True,
                'n_jobs'      : -1,
                'random_state': 42,
#                'class_weight': 'balanced'
            },
            #skopt
            'search_space': {
                'n_estimators'     : Integer(800, 1600),
                'min_samples_split': Integer(16, 64),
                'min_samples_leaf' : Integer(2, 15),
                'max_leaf_nodes'   : Integer(63, 511),
                'max_depth'        : Integer(10, 16),
            },
        },
    },  #RF

    'ScikitXT': {
        'model': ExtraTreesClassifier,
        'hyperparameter_optimization': {
            #skopt itself
            'search_settings':{
                'n_calls'     : 2,
                'n_inits'     : 1,
                'random_state': 42,
            },
            #cv settings
            'evaluation_settings': {
                'validation'  : 'StratifiedKFold',
                'nr_fold'     : 3,    # CV
                'nr_splits'   : 3,    # Splits
                'train_size'  : 0.5,  # split size
                'valid_size'  : 0.5,  # split size
                'split_seed'  : 538,
                'eval_metric' : 'roc_auc',  # sklearn scorer
            },
            #
            'initialize': {
                'criterion'   : 'entropy', #'gini',
                'n_jobs'      : -1,
                'random_state': 42,
#                'class_weight': 'balanced'
            },
            #skopt
            'search_space': {
                'n_estimators'     : Integer(1000, 2000),
                'min_samples_split': Integer(11, 25),
                'min_samples_leaf' : Integer(2, 10),
                'max_leaf_nodes'   : Integer(255, 765),
                'max_depth'        : Integer(10, 16),
            },
        },
    },  #XT

}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 29 2018

@author: cttsai
"""
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier


# parsing result from './data'
ExternalMetaConfigs = {
    'SampleLGBM': [],
}


StackerConfigs = {
    # meta-stacking from mlxtender and external results
    'stacker': [
        {'name'           : 'LinearStacker',
        'meta_classifier' : SGDClassifier,
        'params'          : {
            'loss'        : 'modified_huber',
            'penalty'     : 'elasticnet',
            'l1_ratio'    : 0.15,
            'n_jobs'      : -1,
            'random_state': 42,
            'max_iter'    : 2000,
        },
        'cv'              : 3,
        'use_features'    : False,
        'stratify'        : True,
        'base_classifiers': [
            {'model' : BaggingClassifier,
             'params': {
                    'base_estimator': 'HuberLR',
                    'n_estimators': 10,
                    'max_samples': 0.9,
                    'max_features': 0.8,
                    'n_jobs': -1,
                    'random_state': 42,
                }
            },
            {'model': XGBClassifier,
            'params'           : {
                'objective'        : 'binary:logistic',
                'booster'          : 'gblinear',
                'n_jobs'           : -1,
                'base_score'       : 0.95,
                'scale_pos_weight' : 1.,
                'learning_rate'    : 0.05,
                'n_estimators'     : 100,
                'reg_alpha'        : 0.025,
                'reg_lambda'       : 1.25,
                'eval_metric'      : 'auc'
                    },
                },
            ],
        },
    ],

    # stacking
    'feature': [
        {'name': 'LinearStacker',
        'meta_classifier' : SGDClassifier,
        'params'          : {
                 'loss'        : 'modified_huber', #'log' or 'modified_huber'
                 'penalty'     : 'elasticnet',
                 'l1_ratio'    : 0.15,
                 'n_jobs'      : -1,
                 'random_state': 42,
                 'max_iter'    : 1000},
         'cv'              : 3,
         'seed'            : 538,
         'use_features'    : False,
         'stratify'        : True,
         #'sources': ['RF', 'XT', 'ScikitRF'],
         'sources': ['histXGB', 'LGBM'],
        },
    ],
}


# base classifiers to do stacking and re-stacking
BaseModelConfigs = {

    'GaussianNB': {
        'model' : GaussianNB,
        'params': {},
    },

    'HuberLR': {
        'model': SGDClassifier,
        'params': {
            'loss': 'modified_huber',
            'penalty': 'elasticnet',
            'l1_ratio': 0.15,
            'max_iter': 2000,
            'n_jobs': -1,
            'random_state': 42,
        },
    },

    'wHuberLR': {
        'model': SGDClassifier,
        'params': {
            'loss': 'modified_huber',
            'penalty': 'elasticnet',
            'l1_ratio': 0.15,
            'max_iter': 2000,
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        },
    },

    'LR': {
        'model': SGDClassifier,
        'params': {
            'loss': 'log',
            'penalty': 'elasticnet',
            'l1_ratio': 0.15,
            'max_iter': 2000,
            'n_jobs': -1,
            'random_state': 42,
        },
    },

    'wScikitRF': {
        'model' : RandomForestClassifier,
        'params': {  # override by task
            'n_estimators'     : 10,
            'criterion'        : 'gini',
            'max_depth'        : 12,
            'min_samples_split': 32,
            'min_samples_leaf' : 32,
            'oob_score'        : True,
            'n_jobs'           : -1,
            'random_state'     : 42,
            'class_weight': 'balanced'
        },
    },

    'kNN13': {
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors': 13,
            'weights'    : 'distance', #'uniform',
            'p'          : 1,
            'n_jobs'     :-1,
        },
    },

    'LinearGBM': {
        'model': XGBClassifier,
        'params': {
            'objective'       : 'binary:logistic',
            'booster'         : 'gblinear',
            'n_jobs'          : -1,
            'base_score'      : 0.95,
            'scale_pos_weight': 1,
            'learning_rate'   : 0.05,
            'n_estimators'    : 100,
            'reg_alpha'       : 0.002,
            'reg_lambda'      : 1.,
            'eval_metric'     : 'auc',
        },
    },

    'RF': {'model' : RandomForestClassifier,
           'task': 'ScikitRF',  # load results from HPOs
     },

    'XT': {'model' : ExtraTreesClassifier,
           'task': 'ScikitXT',
    },

    'LGBM': {'model' : LGBMClassifier,
           'task': 'LGBM',  # load results from HPOs
     },

    'histXGB': {
        'model': XGBClassifier,
        'task': 'LossGuideXGB'
    },

}

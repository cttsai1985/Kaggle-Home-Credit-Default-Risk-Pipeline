#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file keep

Created on Tue July 10 2018

This config is original portinr features from and slightly modified
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
https://www.kaggle.com/dromosys/fork-of-fork-lightgbm-with-simple-features-cee847

@author: cttsai
"""

import numpy as np

from sklearn.decomposition import NMF, TruncatedSVD, IncrementalPCA, LatentDirichletAllocation
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

DataConfigs = {
    # input file
    'input' : {
        'application_train'    : {'name': 'application_train.csv',     'index': 'SK_ID_CURR', },
        'application_test'     : {'name': 'application_test.csv',      'index': 'SK_ID_CURR',},
        'previous_application' : {'name': 'previous_application.csv',  'index': 'SK_ID_PREV',},
        'credit_card_balance'  : {'name': 'credit_card_balance.csv',   'index': 'SK_ID_PREV',},
        'pos_cash_balance'     : {'name': 'POS_CASH_balance.csv',      'index': 'SK_ID_PREV',},
        'installments_payments': {'name': 'installments_payments.csv', 'index': 'SK_ID_PREV',},
        'bureau'               : {'name': 'bureau.csv',                'index': 'SK_ID_BUREAU',},
        'bureau_balance'       : {'name': 'bureau_balance.csv',        'index': 'SK_ID_BUREAU',},
    },

    # application train and test
    'application': {
        'filter_rows': {'CODE_GENDER': ['XNA'], },  # dict to feed in pandas directly
        'factorize_columns': ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'],
        'onehot_encoding': True,
        'onehot_columns': [],  # [] mean auto detect
        'nan_as_category': True,
        'replace_rows': {
                'DAYS_BIRTH': {365243: np.nan, },
                'DAYS_EMPLOYED': {365243: np.nan, },
                'DAYS_ID_PUBLISH': {365243: np.nan, },
                'DAYS_REGISTRATION': {365243: np.nan, },
            },
        'interaction_columns': [
            {'name': 'REGION_RATING_CLIENT_RATIO', 'mode': 'divide',   'a': 'REGION_RATING_CLIENT_W_CITY', 'b': 'REGION_RATING_CLIENT',},
            {'name': 'REGION_RATING_CLIENT_MULTI', 'mode': 'multiply', 'a': 'REGION_RATING_CLIENT_W_CITY', 'b': 'REGION_RATING_CLIENT',},
            {'name': 'DEPENDENT_FAM_MEM_RATIO',    'mode': 'divide',   'a': 'CNT_CHILDREN',     'b': 'CNT_FAM_MEMBERS',},
            {'name': 'ADULT_FAM_MEMBERS',          'mode': 'subtract', 'a': 'CNT_CHILDREN',     'b': 'CNT_FAM_MEMBERS',},
            {'name': 'INCOME_PER_PERSON',          'mode': 'divide',   'a': 'AMT_INCOME_TOTAL', 'b': 'CNT_FAM_MEMBERS',},
            {'name': 'INCOME_PER_CHILD',           'mode': 'divide',   'a': 'AMT_INCOME_TOTAL', 'b': 'CNT_CHILDREN',},
            # amount
            {'name': 'CREDIT_TO_ANNUITY_RATIO',    'mode': 'divide',   'a': 'AMT_CREDIT',       'b': 'AMT_ANNUITY',},
            {'name': 'CREDIT_TO_GOODS_RATIO',      'mode': 'divide',   'a': 'AMT_CREDIT',       'b': 'AMT_GOODS_PRICE',},
            {'name': 'CREDIT_TO_INCOME_RATIO',     'mode': 'divide_nonzero', 'a': 'AMT_CREDIT', 'b': 'AMT_INCOME_TOTAL',},
            {'name': 'ANNUITY_TO_INCOME_RATIO',    'mode': 'divide_nonzero', 'a': 'AMT_ANNUITY', 'b' :'AMT_INCOME_TOTAL',},
            {'name': 'CREDIT_MULTI_INCOME',        'mode': 'multiply', 'a': 'AMT_CREDIT', 'b': 'AMT_INCOME_TOTAL',},
            # days
#            {'name': 'EMPLOY_TO_BIRTH_RATIO',      'mode': 'divide',   'a': 'DAYS_EMPLOYED',    'b': 'DAYS_BIRTH',},
#            {'name': 'REGIST_TO_BIRTH_RATIO',      'mode': 'divide',   'a': 'DAYS_REGISTRATION','b': 'DAYS_BIRTH',},
#            {'name': 'ID_PUBLISH_TO_BIRTH_RATIO',  'mode': 'divide',   'a': 'DAYS_ID_PUBLISH',  'b': 'DAYS_BIRTH',},

            {'name': 'CAR_TO_BIRTH_RATIO',         'mode': 'divide',   'a': 'OWN_CAR_AGE',      'b': 'DAYS_BIRTH',},
            {'name': 'CAR_TO_EMPLOY_RATIO',        'mode': 'divide',   'a': 'OWN_CAR_AGE',      'b': 'DAYS_EMPLOYED',},
            {'name': 'PHONE_TO_BIRTH_RATIO',       'mode': 'divide',   'a': 'DAYS_LAST_PHONE_CHANGE', 'b': 'DAYS_BIRTH',},
            {'name': 'PHONE_TO_EMPLOY_RATIO',      'mode': 'divide',   'a': 'DAYS_LAST_PHONE_CHANGE', 'b': 'DAYS_EMPLOYED',},
        ],
        'deep_interactions': [
            {'header'  : 'DOC_IND',
            'transform': ['kurtosis', 'sum', 'mean', 'std'],
            'columns'  : [
                        'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                        'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                        'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',],
            },
            {'header'  : 'EXT_SOURCES_SYNTHESIZE',
            'transform': ['product', 'mean', 'sum', 'sum_squared', 'std'],
            'columns'  : ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
            },
            {'header'  : 'CONTACT_IND',
            'transform': ['kurtosis', 'sum', 'std'],
            'columns'  : [
                        'FLAG_CONT_MOBILE', 'FLAG_MOBIL',
                        'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_EMP_PHONE', 'FLAG_EMAIL',
                        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',],
            },
            {'header'  : 'LIVE_IND',
            'transform': ['kurtosis', 'sum', 'mean', 'std'],
            'columns'  : [
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',],
            },

        ],
        'decomposition': [
                #APPLICATTION and APPLICANT
                {'columns': [
                        'CODE_GENDER',
                        'FLAG_CONT_MOBILE', 'FLAG_MOBIL',
                        'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_EMP_PHONE', 'FLAG_EMAIL',
                        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
                        'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                        'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                        'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',],
                'stems': [
                          'NAME_FAMILY_STATUS_', 'NAME_HOUSING_TYPE_',
                          'NAME_CONTRACT_TYPE_', 'NAME_INCOME_TYPE_',
                          'OCCUPATION_TYPE_', 'ORGANIZATION_TYPE_', 'NAME_EDUCATION_TYPE_'],
                'methods': {'APPLICANT_SVD': {'object': TruncatedSVD,
                                        'params': {'n_components': 4,
                                                   'algorithm': 'randomized',
                                                   'n_iter': 10,
                                                   'random_state': 42},},
                            'APPLICANT_LDA': {'object': LatentDirichletAllocation,
                                                'params': {'n_components': 8,
                                                           'n_jobs':-1,
                                                           'random_state': 42},},
                },
            },
        ]
    },

    'previous_application': {
        'filter_rows': {},
        'factorize_columns': [],
        'onehot_encoding': True,
        'onehot_columns': [],
        'nan_as_category': True,
        'replace_rows': {
            'DAYS_FIRST_DRAWING'       : {365243: np.nan, },
            'DAYS_FIRST_DUE'           : {365243: np.nan, },
            'DAYS_LAST_DUE_1ST_VERSION': {365243: np.nan, },
            'DAYS_LAST_DUE'            : {365243: np.nan, },
            'DAYS_TERMINATION'         : {365243: np.nan, }},
        'interaction_columns': [
            {'name': 'APP_CREDIT_RATIO',  'mode': 'divide',   'a': 'AMT_APPLICATION', 'b':'AMT_CREDIT',},
            {'name': 'APP_CREDIT_DIFF',   'mode': 'subtract', 'a': 'AMT_APPLICATION', 'b':'AMT_CREDIT',},
            {'name': 'EQUITY_INIT_RATIO', 'mode': 'divide',   'a': 'AMT_APPLICATION', 'b':'AMT_CREDIT',},
            {'name': 'EQUITY_DIFF',       'mode': 'subtract', 'a': 'AMT_APPLICATION', 'b':'AMT_CREDIT',},
        ],
        'aggregations': [ # list of aggregation task
            {'header' : "PREV",
             'data'   : 'previous_application',
             'groupby': ['SK_ID_CURR'], #
             'index'  : 'SK_ID_CURR',
             'cat'    : ['mean'], #
             'num'    : {  #
                         'AMT_ANNUITY': ['min', 'max', 'mean'],
                         'AMT_APPLICATION': ['min', 'max', 'mean'],
                         'AMT_CREDIT': ['min', 'max', 'mean'],
                         'APP_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
                         'APP_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
                         'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
                         'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
                         'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                         'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
                         'DAYS_DECISION': ['min', 'max', 'mean', 'var'],
                         'CNT_PAYMENT': ['mean', 'sum'],
                },
            },
            {'header' : "PREV_APPROVED",
             'data'   : 'previous_application',
             'groupby': ['SK_ID_CURR'], #
             'subset' : {'column_name': 'NAME_CONTRACT_STATUS_Approved',
                        'conditions' : [1],},
             'index'  : 'SK_ID_CURR',
             'cat'    : [],
             'num'    : {
                         'AMT_ANNUITY': ['min', 'max', 'mean'],
                         'AMT_APPLICATION': ['min', 'max', 'mean'],
                         'AMT_CREDIT': ['min', 'max', 'mean'],
                         'APP_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
                         'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
                         'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
                         'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                         'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
                         'DAYS_DECISION': ['min', 'max', 'mean'],
                         'CNT_PAYMENT': ['mean', 'sum'],
                },
            },
            {'header': "PREV_REFUSED",
             'data'  : 'previous_application',
             'groupby': ['SK_ID_CURR'],
             'subset': {'column_name': 'NAME_CONTRACT_STATUS_Refused',
                        'conditions' : [1],},
             'index' : 'SK_ID_CURR',
             'cat'   : [],
             'num'   : {
                         'AMT_ANNUITY': ['min', 'max', 'mean'],
                         'AMT_APPLICATION': ['min', 'max', 'mean'],
                         'AMT_CREDIT': ['min', 'max', 'mean'],
                         'APP_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
                         'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
                         'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
                         'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                         'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
                         'DAYS_DECISION': ['min', 'max', 'mean'],
                         'CNT_PAYMENT': ['mean', 'sum'],
                },
            },
        ],
    },

    'bureau':{
        'filter_rows': {},
        'factorize_columns': [],
        'onehot_encoding': True,
        'onehot_columns': [],
        'nan_as_category': True,
        'replace_rows': {},
        'interaction_columns': [],
        'aggregations': [
            {'header': "BB",
             'groupby': ['SK_ID_BUREAU'],
             'index' : 'SK_ID_BUREAU',
             'data'  : 'bureau_balance',
             'count' : True,
             'cat'   : ['mean'], # cat cols by autometically identify
             'num'   : {
                'MONTHS_BALANCE': ['min', 'max', 'size'],
                },
            },
            {'header': "BUREAU",
             'groupby': ['SK_ID_CURR'],
             'index' : 'SK_ID_CURR',#'SK_ID_BUREAU',
             'data'  : 'bureau',
             'cat'   : ['mean'], # cat cols by autometically identify
             'num'   : {
                     'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                     'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                     'DAYS_CREDIT_UPDATE': ['mean'],
                     'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                     'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
                     'AMT_ANNUITY': ['max', 'mean'],
                     'CNT_CREDIT_PROLONG': ['sum'],
                     'BB_MONTHS_BALANCE_MIN': ['min'],
                     'BB_MONTHS_BALANCE_MAX': ['max'],
                     'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],
                     'BB_STATUS_C_MEAN': ['mean', 'sum'],
                     'BB_STATUS_X_MEAN': ['mean', 'sum'],
                     'BB_STATUS_0_MEAN': ['mean', 'sum'],
                     'BB_STATUS_1_MEAN': ['mean', 'sum'],
                     'BB_STATUS_2_MEAN': ['mean', 'sum'],
                     'BB_STATUS_3_MEAN': ['mean', 'sum'],
                     'BB_STATUS_4_MEAN': ['mean', 'sum'],
                     'BB_STATUS_5_MEAN': ['mean', 'sum'],
                },
            },
            {'header' : "BUREAU_ACTIVED",
             'data'   : 'bureau',
             'groupby': ['SK_ID_CURR'],
             'subset' : {'column_name': 'CREDIT_ACTIVE_Active',
                        'conditions' : [1],},
             'index'  : 'SK_ID_CURR',
             'count'  : True,
             'cat'    : [],
             'num'    : {
                     'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                     'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                     'DAYS_CREDIT_UPDATE': ['mean'],
                     'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                     'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
                     'AMT_ANNUITY': ['max', 'mean'],
                     'CNT_CREDIT_PROLONG': ['sum'],
                     'BB_MONTHS_BALANCE_MIN': ['min'],
                     'BB_MONTHS_BALANCE_MAX': ['max'],
                     'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],
                        },
            },
            {'header' : "BUREAU_CLOSED",
             'data'   : 'bureau',
             'groupby': ['SK_ID_CURR'],
             'subset' : {'column_name': 'CREDIT_ACTIVE_Closed',
                        'conditions' : [1],},
             'index'  : 'SK_ID_CURR',
             'count'  : True,
             'cat'    : [],
             'num'    : {
                     'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                     'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                     'DAYS_CREDIT_UPDATE': ['mean'],
                     'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                     'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
                     'AMT_ANNUITY': ['max', 'mean'],
                     'CNT_CREDIT_PROLONG': ['sum'],
                     'BB_MONTHS_BALANCE_MIN': ['min'],
                     'BB_MONTHS_BALANCE_MAX': ['max'],
                     'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],
                        },
            },
            {'header' : "BUREAU_CREDIT_TYPE",
             'data'   : 'bureau',
             'groupby': ['SK_ID_CURR'],
             'subset' : {'column_name': 'BUREAU_CREDIT_TYPE_Consumer credit',
                        'conditions' : [1],},
             'index'  : 'SK_ID_CURR',
             'count'  : True,
             'cat'    : [],
             'num'    : {
                     'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                     'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                     'DAYS_CREDIT_UPDATE': ['mean'],
                     'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                     'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                     'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
                     'AMT_ANNUITY': ['max', 'mean'],
                     'CNT_CREDIT_PROLONG': ['sum'],
                     'BB_MONTHS_BALANCE_MIN': ['min'],
                     'BB_MONTHS_BALANCE_MAX': ['max'],
                     'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],
                        },
            },
        ],
    },

    'installments_payments':{
        'filter_rows': {},
        'factorize_columns': [],
        'onehot_encoding': True,
        'onehot_columns': [],
        'nan_as_category': True,
        'replace_rows': {},
        'interaction_columns': [
            {'name': 'PAYMENT_RATIO','mode': 'divide',   'a': 'AMT_PAYMENT', 'b':'AMT_INSTALMENT',},
            {'name': 'PAYMENT_DIFF', 'mode': 'subtract', 'a': 'AMT_INSTALMENT', 'b':'AMT_PAYMENT',},
            {'name': 'DPD',          'mode': 'subtract_positive', 'a': 'DAYS_ENTRY_PAYMENT', 'b':'DAYS_INSTALMENT',},
            {'name': 'DBD',          'mode': 'subtract_positive', 'a': 'DAYS_INSTALMENT', 'b':'DAYS_ENTRY_PAYMENT',},
        ],
        'aggregations':[
            {'header' : "INSTALL",
             'groupby': ['SK_ID_CURR'],
             'data'   : 'installments_payments',
             'index'  : 'SK_ID_CURR',
             'count'  : True,
#             'cat'    : ['mean'], # cat cols by autometically identify
             'num'    : {
                     'NUM_INSTALMENT_VERSION': ['nunique'],
                     'DPD': ['max', 'mean', 'sum'],
                     'DBD': ['max', 'mean', 'sum'],
                     'PAYMENT_RATIO': ['max', 'mean', 'sum', 'var'],
                     'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
                     'AMT_INSTALMENT': ['max', 'mean', 'sum'],
                     'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
                     'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
                     },
            },
        ],
    },

    'pos_cash': {
        'filter_rows': {},
        'factorize_columns': [],
        'onehot_encoding': True,
        'onehot_columns': [],
        'nan_as_category': True,
        'replace_rows': {},
        'interaction_columns': [
            {'name': 'APP_CREDIT_RATIO', 'mode': 'divide', 'a': 'AMT_APPLICATION', 'b':'AMT_CREDIT',},
        ],
        'aggregations': [
                {'header': "POS_CASH",
                'groupby': ['SK_ID_CURR'],
                 'data'  : 'pos_cash_balance',
                 'index' : 'SK_ID_CURR',
                 'count' : True,
                 'cat'   : ['mean'], # cat cols by autometically identify
                 'num'   : {
                    'MONTHS_BALANCE': ['max', 'mean', 'size'],
                    'SK_DPD'        : ['max', 'mean', 'var'],
                    'SK_DPD_DEF'    : ['max', 'mean', 'var'],
                },
            },
        ],
        'conditional_aggregations': [], # omit aggregations
    },

    'credit_card_balance': {
        'filter_rows': {},
        'factorize_columns': [],
        'onehot_encoding': True,
        'onehot_columns': [],
        'nan_as_category': True,
        'replace_rows': {},
        'interaction_columns': [
            {'name': 'APP_CREDIT_RATIO', 'mode': 'divide', 'a': 'AMT_APPLICATION', 'b':'AMT_CREDIT',},
        ],
        'aggregations': [ # general aggregation
            {'header' : "CC",
             'groupby': ['SK_ID_CURR'],
             'data'   : 'credit_card_balance',
             'index'  : 'SK_ID_CURR',
             'count'  : True,
             'cat'    : ['mean', 'sum'],
             'num'    : {
                'MONTHS_BALANCE'            : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_BALANCE'               : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_CREDIT_LIMIT_ACTUAL'   : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_DRAWINGS_ATM_CURRENT'  : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_DRAWINGS_CURRENT'      : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_DRAWINGS_POS_CURRENT'  : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_INST_MIN_REGULARITY'   : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_PAYMENT_CURRENT'       : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_PAYMENT_TOTAL_CURRENT' : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_RECEIVABLE_PRINCIPAL'  : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_RECIVABLE'             : ['min', 'max', 'mean', 'sum', 'var'],
                'AMT_TOTAL_RECEIVABLE'      : ['min', 'max', 'mean', 'sum', 'var'],
                'CNT_DRAWINGS_ATM_CURRENT'  : ['min', 'max', 'mean', 'sum', 'var'],
                'CNT_DRAWINGS_CURRENT'      : ['min', 'max', 'mean', 'sum', 'var'],
                'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum', 'var'],
                'CNT_DRAWINGS_POS_CURRENT'  : ['min', 'max', 'mean', 'sum', 'var'],
                'CNT_INSTALMENT_MATURE_CUM' : ['min', 'max', 'mean', 'sum', 'var'],
                'SK_DPD'                    : ['min', 'max', 'mean', 'sum', 'var'],
                'SK_DPD_DEF'                : ['min', 'max', 'mean', 'sum', 'var'],
                }
            },
        ],
    },
}

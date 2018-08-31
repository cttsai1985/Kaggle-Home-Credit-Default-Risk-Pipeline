#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides feature tranform and forked from
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
Created on Thu July 20 2018

@author: cttsai
"""
import gc; gc.enable()
import itertools

import numpy as np
import pandas as pd

from LibConfigs import logger
from Utility import IdentifyCategoricalColumn, CheckColumnsExist


def process_one_hot_encode(df, categorical_columns=[], nan_as_category=True):
    """
    ------
    return df, new_columns, columns_to_convert
    """
    logger.info("Process OneHot Encoding")
    original_columns = df.columns.tolist()

    if not categorical_columns:
        categorical_columns = IdentifyCategoricalColumn(df)
    categorical_columns, _ = CheckColumnsExist(df, categorical_columns)

    logger.info("identify {} categorical columns: {}".format(len(categorical_columns), categorical_columns))
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]
    logger.info("one-hot encoded to {} columns:".format(len(new_columns)))
    df[new_columns] = df[new_columns].astype(np.int8)
    ret = {cat: sorted([col for col in new_columns if cat in col]) for cat in categorical_columns}
    for k, v in ret.items():
        logger.info("onehot {} to {} columns: {}".format(k, len(v), v))

    return df, new_columns, categorical_columns


def process_interaction(df, process_configs):
    """
    process configs is a dictionary as
    a dictionary with {'new_feature_name': {'mode': 'add', 'a': 'col_name', 'b':'col_name',}, }
    ------

    """
    logger.info("Process Interactions")

    possible_arithmetics = ['add', 'sum_squared',
                            'subtract', 'subtract_positive',
                            'multiply',
                            'divide', 'divide_nonzero']

    new_columns = []
    for v in process_configs:
        k = v['name']
        logger.info("process {}".format(k))

        # check arithmetic
        arithmetic = v.get('mode', None)
        if arithmetic not in possible_arithmetics:
            logger.warning("no arithmetic on {}".format(k))
            continue

        #check feature columns
        ckeck_cols = [vv for kk, vv in v.items() if kk not in ['name', 'mode']]
        cols_exist, cols_not_exist = CheckColumnsExist(df, ckeck_cols)
        if cols_not_exist:
            logger.warning("missing {} columns: {}".format(len(cols_not_exist), cols_not_exist))
            continue

        # process
        if 'add' == arithmetic:
            df[k] = df[v['a']] + df[v['b']]
        elif 'subtract' == arithmetic:
            df[k] = df[v['a']] - df[v['b']]
        elif 'subtract_positive' == arithmetic:
            df[k] = (df[v['a']] - df[v['b']]).apply(lambda x: x if x > 0 else 0)
        elif 'multiply' == arithmetic:
            df[k] = df[v['a']] * df[v['b']]
        elif 'divide' == arithmetic:
            df[k] = df[v['a']] / df[v['b']]
        elif 'divide_nonzero' == arithmetic:
            df[k] = df[v['a']] / (df[v['b']] + 1.)
        elif 'sum_squared' == arithmetic:
            df[k] = df[[v['a'], v['b']]].pow(2).sum(axis=1)# np.square(df[v['a']]) + np.square(df[v['b']])

        new_columns.append(k)

    return df, new_columns


def process_deep_interactions(df, process_configs):
    """
    {'header'   : 'EXT_SOURCES_SYNTHESIZE',
     'transform': ['product', 'mean', 'sum', 'sum_squared', 'std'],
     'columns'  : ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
     }
    """
    applicable_methods = ['kurtosis', 'sum', 'sum_squared', 'product', 'mean', 'std']

    header  = process_configs.get('header', 'NEW')
    cols    = process_configs.get('columns', [])
    cols_na = [f for f in cols if f not in df.columns]
    cols    = [f for f in cols if f in df.columns]
    methods = process_configs.get('transform', [])
    methods = [m for m in methods if m in applicable_methods]

    for m in methods:
        logger.info('transform deep interactions ({}): {}'.format(m, cols))
        if cols_na:
            logger.warning('transform deep interactions ({}), features not found: {}'.format(
                    m, cols_na))

        name = '{}_{}'.format(header, m.upper())
        if m == 'kurtosis':
            df[name] = df[cols].kurtosis(axis=1)
        elif m == 'mean':
            df[name] = df[cols].mean(axis=1)
        elif m == 'sum':
            df[name] = df[cols].sum(axis=1)
        elif m == 'sum_squared':
            df[name] = df[cols].pow(2).sum(axis=1)
        elif m == 'product':
            df[name] = df[cols].fillna(df[cols].mean()).product(axis=1)
        elif m == 'std':
            df[name] = df[cols].std(axis=1)
            df[name] = df[name].fillna(df[name].mean())

    return df


def process_replace(df, process_configs):
    """
    {'DAYS_EMPLOYED': {365243: np.nan, }, }
    """
    logger.info("Process Fill NA")
    columns = sorted(list(process_configs.keys()))
    cols_exist, cols_not_exist = CheckColumnsExist(df, columns)

    configs = {k: v for k, v in process_configs.items() if k in cols_exist}
    df.replace(configs, inplace=True)

    for k, v in configs.items():
        logger.info("impute {} using {}".format(k, v))
    if cols_not_exist:
        logger.warning("missing {} columns: {}".format(len(cols_not_exist), cols_not_exist))

    return df


def process_drop_rows(df, process_configs):
    """
    {'CODE_GENDER': ['XNA'], }
    """
    logger.info("Process Drop Rows")
    columns = sorted(list(process_configs.keys()))
    cols_exist, cols_not_exist = CheckColumnsExist(df, columns)

    configs = {k: v for k, v in process_configs.items() if k in cols_exist}
    inds = df[cols_exist].isin(configs)
    inds_sel = inds.any(axis=1)

    for f, series in inds.iteritems():
        logger.info("remove {} rows by in {} if any {}".format(f, series.sum(), process_configs[f]))

    logger.info("overall remove {} from {} rows".format(inds_sel.astype(int).sum(), inds_sel.shape[0]))
    if cols_not_exist:
        logger.warning("missing {} columns: {}".format(len(cols_not_exist), cols_not_exist))

    return df.loc[~inds_sel]


def process_factorize(df, process_configs):
    """
    input a list of features to factorize (label encoding)
    """
    logger.info("Process Factorize")
    cols_exist, cols_not_exist = CheckColumnsExist(df, sorted(process_configs))

    for bin_feature in cols_exist:
        df[bin_feature], uniques = pd.factorize(df[bin_feature], sort=False)
        logger.info("factorize {} in {}: {}".format(len(uniques), bin_feature, uniques))

    for k in cols_not_exist:
        logger.warning("missing {}".format(k))

    return df


def process_aggregate(df, process_configs, groupby_cols, cat_cols=[]):
    """
    pass each groupby_cols one by one: aggregate and condictional aggregate, general aggregate
    """
    logger.info("Process Aggregate")
    groupby_cols = [f for f in groupby_cols if f in df.columns]
#    if groupby_cols not in df.columns:
    if not groupby_cols:
        logger.warning("aggregate column {} not exist".format(groupby_cols))
        return pd.DataFrame({groupby_cols:[]}).set_index(groupby_cols)

    logger.info("aggregate on {}".format(groupby_cols))
    header = process_configs.get('header', 'foobar')

    aggregations = {}
    # aggregate and condictional aggregate
    num_cols = process_configs.get('num', {})
    cat_agg = process_configs.get('cat', [])
    if num_cols or cat_agg:
        aggregations = {k:list(v) for k, v in num_cols.items() if k in df.columns and v}
        aggregations.update({k:list(cat_agg) for k in cat_cols if k in df.columns and cat_agg})
        for k, v in aggregations.items():  # dict
            logger.info("aggregate {} ({}) with {}".format(k, df[k].dtype, v))

        # assigned in configs but not in dataframe
        missing = sorted(list(set(num_cols.keys()).union(set(cat_cols)).difference(set(aggregations.keys()))))
        for k in missing:  # dict
            if k in num_cols.keys():
                logger.info("missing {} in num".format(k))
            elif k in cat_cols:
                logger.info("missing {} in cat".format(k))

    #  processing
    if aggregations:
        df_agg = df.groupby(groupby_cols).agg({**aggregations})
        df_agg.columns = pd.Index(['{}_{}_{}'.format(header, e[0], e[1].upper()) for e in df_agg.columns.tolist()])
    else:
        logger.info("no aggragation on {} and {}".format(header, groupby_cols))
        df_agg = pd.DataFrame({groupby_cols:[]}).set_index(groupby_cols)

    if process_configs.get('count', False):
        logger.info("aggregate count on {} at {}".format(groupby_cols, header))
        df_agg['{}_COUNT_{}'.format(header, '_'.join(groupby_cols))] = df.groupby(groupby_cols).size()

    return df_agg


def process_decomposition(df, process_configs):
    """
    {'columns': ['FLAG_CONT_MOBILE', 'FLAG_PHONE'],
    'stems'   : ['CODE_GENDER_'],
    'methods' : {'APPLICANT_SVD': {'object': TruncatedSVD,
                                       'params': {'n_components': 8,
                                                   'algorithm': 'randomized',
                                                   'n_iter': 10,
                                                   'random_state': 42},},
        },
    }
    """
    use_cols, cols_not_exist = CheckColumnsExist(df, process_configs.get('columns', []))
    stems = process_configs.get('stems', [])
    if stems:
        dict_stem = {s:[f for f in df.columns if s in f] for s in stems}
        cols_stem = list(itertools.chain.from_iterable(dict_stem.values()))
        if cols_stem:
            use_cols.extend(cols_stem)
            for k, v in dict_stem.items():
                logger.info('find {} stem "{}": {}'.format(len(v), k, v))

    use_cols = sorted(use_cols)
    logger.info('decompose on {} features: {}'.format(len(use_cols), use_cols))
    df_sub = df[use_cols].apply(lambda x: np.nan_to_num(x))

    def func(k, v, sub):
        tf     = v.get('object', None)
        params = v.get('params', {})
        if not tf:
            return pd.DataFrame()
        logger.info('decompose {} on {} features'.format(k, sub.shape[1]))
        d = tf().set_params(**params).fit_transform(sub)
        return pd.DataFrame(d, columns=['{}_{}'.format(k, i) for i in range(1, d.shape[1]+1)])

    ret = [func(k, v, df_sub) for k, v in process_configs.get('methods', {}).items()]
    ret = pd.concat(ret, axis=1, join='inner')
    ret.index = df.index
    return ret

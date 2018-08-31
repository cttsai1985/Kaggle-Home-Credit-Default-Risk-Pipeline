#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 2018

@author: cttsai
"""
import os, sys
import pickle
from datetime import datetime as dt

import numpy as np
import pandas as pd

from LibConfigs import logger
from LibConfigs import enable_gpu_options, disable_gpu_options


def SwitchDevice(params, enable_gpu=True):

    def func(params, opt, silent=False):
        for k, v in params.items():
            a = k in opt.keys()
            b = v in opt.get(k, {}).keys()
            if all([a, b]):
                params.update({k: opt[k].get(v)})
                logger.info('switch {} to {}'.format(k, params[k]))
        return params

    switched_params = params.copy()
    if enable_gpu:
        switched_params = func(switched_params, enable_gpu_options)
    else:
        switched_params = func(switched_params, disable_gpu_options)

    return switched_params


def IdentifyCategoricalColumn(df):
    return [col for col in df.columns if df[col].dtype == 'object']


def CheckColumnsExist(df, columns):
    cols_exist = [f for f in columns if f in df.columns]
    cols_not_exist = [f for f in columns if f not in df.columns]
    return cols_exist, cols_not_exist

def CheckFileExist(filename, silent=True):
    if not os.path.exists(filename):
        if not silent:
            logger.warning('{} does not exist'.format(filename))

        return False

    return True


def MkDirSafe(directory):
    if not os.path.exists(directory):
        logger.info('make {}'.format(directory))
        os.makedirs(directory)


def AnyEmptyDataframe(data):
    if not data:
        logger.warning('passing no dataframes')
        return True

    if isinstance(data, dict):
        return any([v.empty for k, v in data.items()])

    elif isinstance(data, list):
        return any([l.empty for l in data])

    return False


def SavePickle(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        logger.info('save model to {}'.format(filename))


def LoadPickle(filename):
    if not CheckFileExist(filename):
        return None

    with open(filename, 'rb') as f:
        logger.info('load model {}'.format(filename))
        return pickle.load(f)


def Cast64To32(df, blacklist=['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']):
    series_dtypes = df.dtypes
    series_dtypes = series_dtypes.loc[~series_dtypes.index.isin(blacklist)]
    if not series_dtypes.empty:
        logger.info('cast dataframe from 64 to 32')
        to_float32 = series_dtypes.loc[series_dtypes.apply(lambda x: x == np.float64)].index.tolist()
        df[to_float32] = df[to_float32].astype(np.float32)
        logger.info('cast {} columns float32: {}'.format(len(to_float32), to_float32))

        to_int32 = series_dtypes.loc[series_dtypes.apply(lambda x: x == np.int64)].index.tolist()
        df[to_int32] = df[to_int32].astype(np.int32)
        logger.info('cast {} columns to int32: {}'.format(len(to_int32), to_int32))

    return df


def InitializeConfigs(filename):
    if not filename:
        return None

    if not os.path.exists(filename):
        raise ValueError("Spec file {spec_file} does not exist".format(spec_file=filename))

    module_name = filename.split(os.sep)[-1].replace('.', '')

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ComposeResultName(meta={}):
    """
    input
    ----
    meta = {
    'level': int,
    'model': str,
    'feature_num': int,
    'score': float,
    'fold': int,
    'timestamp': datetime.datetime}

    return
    ----
    [level]_[model_type]_[feature_num]_[local_score]_[local_cv]_[time]

    """
    #logger.info('{}'.format(meta))
    template = 'level[{level}]_{model_type}_features[{feature_num:04d}]_score[{score:.6f}]_fold[{fold}]_{timestamp}'
    result_name = template.format(level=int(meta.get('level', 0)),
                                  model_type=meta.get('model', 'unknown'),
                                  feature_num=meta.get('feature_num', 0),
                                  score=meta.get('score', 0.0),
                                  fold=int(meta.get('fold', 0)),
                                  timestamp=meta.get('timestamp', dt.now()).strftime('%Y-%m-%d-%H-%M'))
    return result_name


def DecomposeResultName(name):
    """
    input
    ----
    [level]_[model_type]_[feature_num]_[local_score]_[local_cv]_[time] or
    'header' + [level]_[model_type]_[feature_num]_[local_score]_[local_cv]_[time].extinsion

    return
    ----
    meta = {
    'level': int,
    'model': str,
    'feature_num': int,
    'score': float,
    'fold': int,
    'timestamp': datetime.datetime}
    """
    if (len(name) - name.rfind('.') < 10):  #have an extension
        name = name[name.find('level'):name.rfind('.')]
    else:
        name = name[name.find('level'):]

    level, model_type, feature_num, score, nr_fold, timestamp = name.split('_')

    def extract(x):
        return x.split(']')[0].split('[')[1]

    return {'level': int(extract(level)),
            'model': model_type,
            'feature_num': int(extract(feature_num)),
            'score': float(extract(score)),
            'fold': int(extract(nr_fold)),
            'timestamp': dt.strptime(timestamp, '%Y-%m-%d-%H-%M')}


###############################################################################
def main(argc, args):
    """
    this is a testing module
    """
    print('Test Compose Result Name')
    print(ComposeResultName({}))

    n = ComposeResultName({'level': 1, 'model': 'xgb', 'score': 1.4, 'fold': 3})
    print(n, DecomposeResultName(n))

    n = 'subm_' + n + '.csv'
    print(n, DecomposeResultName(n))

    import pdb; pdb.set_trace()

    return

if __name__ == '__main__':
    main(len(sys.argv), sys.argv)


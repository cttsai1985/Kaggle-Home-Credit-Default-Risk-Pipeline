#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file keep all the configs associated in lib folder.

Created on Tue July 10 2018

@author: cttsai
"""
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit

import logging  # the default logging format
formatter = '%(asctime)s %(filename)s:%(lineno)d: %(message)s'
logging.basicConfig(format=formatter, level='INFO')
logger = logging.getLogger(__name__)

#import pandas as pd
#pd.set_option('display.height', 2000)
#pd.set_option('display.max_rows', 2000)

file_dir_path = {
    'data'    : './data',
    'params'  : './params',
    'configs' : './configs',
    'output'  : './output',
}

hdf5_compress_option = {
    'complevel' : 5,
    'complib'   : 'zlib',
}

fast_hdf5_compress_option = {
    'complevel' : 3,
    'complib'   : 'zlib',
}

data_provider_refresh_configs = {
    'from_csv'        : {'level': 0, 'filename': None},
    'from_raw_cache'  : {'level': 1, 'filename': 'cache_{header}_raw.hdf5'},
    'from_processed'  : {'level': 2, 'filename': 'cache_{header}_processed.hdf5'},
    'from_train_test' : {'level': 3, 'filename': 'cache_{header}_train_test.hdf5'},
}

model_selection_object = {
    'KFold'                  : KFold,
    'StratifiedKFold'        : StratifiedKFold,
    'ShuffleSplit'           : ShuffleSplit,
    'StratifiedShuffleSplit' : StratifiedShuffleSplit,
    'TimeSeriesSplit'        : TimeSeriesSplit,
}


# this enable dict() is pairing with the diable one to open and close gpu related paramters
enable_gpu_options = {
    'device'     : {'cpu': 'gpu'},
    'tree_method': {
        'exact': 'gpu_exact',
        'hist' : 'gpu_hist',
    },
}

disable_gpu_options = {k: {vv: vk for vk, vv in v.items()} for k, v in enable_gpu_options.items()}

#Scikit-Opt
filename_hpo_intermediate = '{loc}/skopt_{prefix}_{stem}_hyperparameters_iter{iter_num:04d}.pk'
filename_hpo_result       = '{loc}/skopt_{prefix}_{stem}_hyperparameters.pk'
filename_hpo_external     = '{loc}/skopt_{prefix}_{task}_hyperparameters.pk'

#stacker
filename_submit_mlxtend_meta   = '{loc}/subm_{prefix}_mlxtend_{stem}_meta.csv'
filename_submit_mlxtend_base   = '{loc}/subm_{prefix}_mlxtend_{stem}_base.csv'
filename_mlxtend_meta_features = '{loc}/{prefix}_mlxtend_{stem}_meta_features.hdf5'
filename_mlxtend_meta_features_external = '{loc}/{prefix}_mlxtend_meta_features.hdf5'
filename_mlxtend_stacker_external = '{loc}/{prefix}_mlxtend_meta_stackers.hdf5'

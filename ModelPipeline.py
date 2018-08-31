#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 29 2018

@author: cttsai
"""

import sys, os
import argparse
import pickle

import ModulePaths
from lib.LibConfigs import logger, file_dir_path
from lib.DataProvider import DataProvider
from lib.ScikitOptimize import ScikitOptimize
from lib.FeatureImportance import FeatureImportance
from lib.AutoStacker import AutoStacker
from lib.Utility import InitializeConfigs, CheckFileExist, SwitchDevice

def parse_command_line():

    default_cache_prefix  = 'sample'

    params_loc  = file_dir_path.get('params', './params')
    configs_loc = file_dir_path.get('configs', './configs')
    default_data_configs_path    = '{}/SampleDataConfigs.py'.format(configs_loc)
    default_model_configs_path   = '{}/SampleModelConfigs.py'.format(configs_loc)
    default_stacker_configs_path = '{}/SampleStackerConfigs.py'.format(configs_loc)
    default_select_to_hpo = None
    default_feature_score_cutoff = 10.

    parser = argparse.ArgumentParser(description='Home Credit Default Risk Modeler',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--cache-prefix',    type=str, default=default_cache_prefix, help='specifiy cache file prefix')
    parser.add_argument('-d', '--configs-data',    type=str, default=default_data_configs_path,  help='path to data configs')
    parser.add_argument('-m', '--configs-model',   type=str, default=default_model_configs_path, help='path to model configs')
    parser.add_argument('-s', '--configs-stacker', type=str, default=default_stacker_configs_path, help='path to stacker configs')
    parser.add_argument('-t', '--select-hpo',      type=str, default=default_select_to_hpo, help='hpo on selected models')
    parser.add_argument(      '--cutoff-score',  type=float, default=default_feature_score_cutoff, help='cutoff to remove unimportant features')
    parser.add_argument('-c', '--cull_features', action='store_true', default=False, help='cull features')
    parser.add_argument('--enable-gpu',    action='store_true', default=False,  help='compute using gpu')
    parser.add_argument('--refresh-cache', action='store_true', default=False,  help='refresh cache by data configs')
    parser.add_argument('--refresh-meta', action='store_true', default=False,  help='refresh constructed meta features')
    parser.add_argument('--compute-hpo',   action='store_true', default=False, help='hpo')
    parser.add_argument('--compute-stack', action='store_true', default=False, help='stacking')
    parser.add_argument('--debug', action='store_true', default=False, help='debug moode using 20000 samples')

    args = parser.parse_args()

    logger.info('running task with prefix={}'.format(args.cache_prefix))

    if args.enable_gpu:
        logger.info('enable GPU computing in hyperparameters')

    if args.cull_features:
        logger.info('cull feature features scores under {}'.format(args.cutoff_score))

    if args.select_hpo:
        args.select_hpo = args.select_hpo.split(',')

    if args.debug:
        logger.warning('**Debug Mode**')
        args.configs_model   = '{}/DebugModelConfigs.py'.format(configs_loc)
        args.configs_stacker = '{}/DebugStackerConfigs.py'.format(configs_loc)

    return args


def compute(args):

    # loading configs
    DataConfigs    = InitializeConfigs(args.configs_data).DataConfigs
    if args.compute_hpo:
        ModelConfigs = InitializeConfigs(args.configs_model).ModelConfigs
    if args.compute_stack:
        StackerConfigs = InitializeConfigs(args.configs_stacker).StackerConfigs
        BaseModelZoo   = InitializeConfigs(args.configs_stacker).BaseModelConfigs
        ExtMetaConfigs = InitializeConfigs(args.configs_stacker).ExternalMetaConfigs


    dp = DataProvider(IOConfigs=file_dir_path)
    if args.refresh_cache:
        data = dp.LoadData(DataConfigs, source='from_processed', prefix=args.cache_prefix)
    else:
        data = dp.LoadData(DataConfigs, source='from_train_test', prefix=args.cache_prefix)

    train_x, train_y, test_x, test_y = data

    if args.cull_features:  # a bit feature selection
        f_path = InitializeConfigs(args.configs_model).fileFeatureImportance
        featSel = FeatureImportance()
        featSel.LoadResult(f_path)
        blacklist = featSel.GetBlacklist(args.cutoff_score)
        train_x = featSel.CullFeatures(train_x, blacklist)
        test_x = featSel.CullFeatures(test_x, blacklist)

    if args.debug:
        train_x = train_x.iloc[:20000]
        train_y = train_y.iloc[:20000]
        logger.warning('debug mode: x={}'.format(train_x.shape))
        args.cache_prefix = 'debug'
    logger.info('P/N ratio:\n{}'.format(train_y.value_counts(normalize=True)))

    if args.compute_hpo:
        logger.info('load hpo configs of {} models'.format(len(ModelConfigs)))
        if args.select_hpo:
            ModelConfigs = {k: v for k, v in ModelConfigs.items() if k in args.select_hpo}
            logger.info('compute hpo for selected {} models'.format(len(ModelConfigs)))

        for k, v in ModelConfigs.items():
            try:
                model      = v.get("model")
                hpo_range  = v.get("hyperparameter_optimization")
                init = hpo_range.get('initialize', {})
                hpo_range.update({'initialize': SwithDevice(init, enable_gpu=args.enable_gpu)})
                hpo_search = ScikitOptimize(model,
                                            hpo_range,
                                            task_name='{}'.format(k),
                                            data_prefix=args.cache_prefix)
                hpo_search.search(train_x, train_y)
                hpo_search.save_hyperparameters(export=True)
                # TODO: fine tune model
            except:
                logger.info('Errors in optimizing {}'.format(task_name='{}'.format(k)))

    if args.compute_stack:
        stackers = AutoStacker(StackerConfigs, args.enable_gpu,
                               data_prefix=args.cache_prefix)

        if args.refresh_meta:
            stackers.buildMetaFeatures(BaseModelZoo)
            stackers.fit_transform(train_x, train_y, test_x, seed=42)

        else:
            stackers.loadExternalMeta(ExtMetaConfigs)
            stackers.buildMetaClassifiers(BaseModelZoo)
            stackers.fit_predict(train_x, train_y, test_x, seed=538)

    return


def main(argc, argv):
    logger.info('reading arguments')
    args = parse_command_line()

    logger.info('starting to compute')
    compute(args)

    return


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

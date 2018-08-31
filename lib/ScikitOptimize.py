#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 26 2018

@author: cttsai
"""

import os, sys
import pickle
import numpy as np
import pandas as pd

from skopt import gp_minimize
from sklearn.model_selection import cross_val_score

from LibConfigs import logger, file_dir_path, model_selection_object
from LibConfigs import filename_hpo_intermediate, filename_hpo_result, filename_hpo_external
from Utility import CheckFileExist, ComposeResultName, SwitchDevice

class ScikitOptimize(object):
    """
    """
    def __init__(self, model, configs={}, task_name=None, data_prefix=None):
        self.model       = model
        self.task_name   = task_name
        self.data_prefix = data_prefix
        self.params_dir  = file_dir_path.get('params', '../params')

        #skopt
        search_settings    = configs.get("search_settings", {})
        self.n_calls       = search_settings.get("n_calls", 15)
        self.random_state  = search_settings.get("random_state", 42)
        self.n_init_points = search_settings.get("n_inits", 10)

        if self.n_init_points >= self.n_calls:
            logger.warning('initial points {} is larger than n_calls {}'.format(self.n_init_points,
                           self.n_calls))

        #validation
        evalute_settings  = configs.get("evaluation_settings", {})

        self.valid_type   = evalute_settings.get("validation", "KFold")
        self.nr_fold      = evalute_settings.get("nr_fold", 3)
        self.split_seed   = evalute_settings.get("split_seed", 42)
        self.metric       = evalute_settings.get("eval_metric", "neg_log_loss")

        #model
        self.init_params     = configs.get("initialize", {})
        self.search_space    = configs.get("search_space", {})
        self.set_params_safe = self._check_parameters()

        self.optimized_params = {}
        self.filename_hpo_iter = ''
        self.filename_hpo_best = ''

        #initializing
        self._search_space_initialize()

        #
        self.filestem_meta = {
            'level': 0,
            'model': self.task_name,
            'feature_num': 0,
            'score': 0,
            'fold': self.nr_fold, }

    def _search_space_initialize(self):
        self.eval_params_name   = sorted([k for k in self.search_space.keys()])
        self.search_params_list = [self.search_space[k] for k in self.eval_params_name]
        logger.info('search range of skopt:')
        for k, v in self.search_space.items():
            logger.info('search {} in {}'.format(k, v))

    def _current_file_stem(self):
        return ComposeResultName(self.filestem_meta)
    #
    def _check_parameters(self):
        m = self.model()
        availabe_params = m.get_params()
        parameters = [k for k in self.init_params.keys()] + [k for k in self.search_space.keys()]
        if any([k not in availabe_params for k in parameters]):
            return False
        else:  # need all parameters in get_params() so safe to call set_params()
            return True

    def get_result_filename(self):
        return self.filename_hpo_best

    def get_optimal_parameters(self):
        if not self.optimized_params:
            logger.warning('need to run optimize first')

        return self.optimized_params.copy()

    def load_hyperparameters(self, filename):
        if not CheckFileExist(filename, silent=False):
            logger.warning('no hpo parameters load from {}'.format(filename))
            return {}

        with open(filename, 'rb') as f:
            params = pickle.load(f)
            logger.info('load from {} with params:{}'.format(filename, params))
            return params

    @staticmethod
    def _save_pickle(filename, obj):
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    def save_hyperparameters(self, export=False, show_iter=True, remove_old=True):
        if not self.optimized_params:
            logger.warning('need to run optimize first')
            return False

        params = SwitchDevice(self.optimized_params, enable_gpu=False)

        if export:
            filename = filename_hpo_external.format(loc=self.params_dir,
                                                    prefix=self.data_prefix,
                                                    task=self.task_name)
            logger.warning('export for external module: {}'.format(filename))
            self._save_pickle(filename, obj=params)
            return filename

        if remove_old and CheckFileExist(self.filename_hpo_best, silent=True):
            os.remove(self.filename_hpo_best)

        stem = self._current_file_stem()
        if show_iter:
            self.filename_hpo_iter = filename_hpo_intermediate.format(loc=self.params_dir,
                                                                      prefix=self.data_prefix,
                                                                      iter_num=self.nr_iteration,
                                                                      stem=stem)
            self._save_pickle(self.filename_hpo_iter, obj=params)

        #write current best anyway
        self.filename_hpo_best = filename_hpo_result.format(loc=self.params_dir,
                                                            prefix=self.data_prefix,
                                                            stem=stem)
        self._save_pickle(self.filename_hpo_best, obj=params)
        #self.load_hyperparameters(filename)  # attemp to reload
        return True

    def _evaluate(self, eval_params):
        eval_params = dict(zip(self.eval_params_name, eval_params))
        tuning_params = self.init_params.copy()
        tuning_params.update(eval_params)

        # reinitialize cv
        cv_obj = self.nr_fold
        if self.valid_type == 'TimeSeriesSplit':
            cv_obj = model_selection_object[self.valid_type](n_splits=self.nr_fold)
        elif 'KFold' in self.valid_type:
            cv_obj = model_selection_object[self.valid_type](n_splits=self.nr_fold,
                                                             shuffle=True,
                                                             random_state=self.split_seed)

        if self.set_params_safe:
            try:
                m = self.model().set_params(**tuning_params)
            except:
                logger.warning('fail to use set_params')
                m = self.model(**tuning_params)
                logger.warning('model params={}'.format(m.get_params()))
        else:  # unless some parameters cannot pass through set_params()
            m = self.model(**tuning_params)

        score  = np.mean(cross_val_score(m,
                                         self.X,
                                         self.y,
                                         cv=cv_obj,
                                         n_jobs=1,
                                         scoring=self.metric))

        self.nr_iteration += 1
        self.best_score = max(self.best_score, score)

        # save the current best paramerters here
        if self.best_score == score:
            # update new result
            self.filestem_meta.update({'score': score})
            self.optimized_params = tuning_params.copy()
            if self.nr_iteration >= self.n_init_points:
                self.save_hyperparameters(show_iter=True)
            else:
                self.save_hyperparameters(show_iter=False)

        if self.nr_iteration == self.n_init_points:  # save after intinializing
            self.save_hyperparameters(show_iter=False)

        logger.info('iteration {:04d}/{:04d}, current score: {:04f}, best: {:.4f}, current params: {}, best params: {}'.format(self.nr_iteration,
                    self.n_calls,
                    score, self.best_score,
                    tuning_params, self.optimized_params))

        return -score # for minimize, most scikit-learn metric are larger the better

    def search(self, X, y):
        self.X = X.apply(lambda x: np.nan_to_num(x))
        self.y = y

        self.filestem_meta.update({'feature_num': X.shape[1],})

        self.nr_iteration = 0
        self.best_score   = 0
        logger.info('evaluate {} at {} iteration, {}-fold cv, metric={}'.format(self.task_name,
                self.n_calls,
                self.nr_fold,
                self.metric))
        gp_optimizer = gp_minimize(self._evaluate,
                                   self.search_params_list,
                                   n_calls=self.n_calls,
                                   n_random_starts=self.n_init_points,
                                   random_state=self.random_state,
                                   verbose=False)

        optimized_params = {k: v for k, v in zip(self.eval_params_name, gp_optimizer.x)}  # not using
        logger.info('best cv score: {}, hyperparameters={}'.format(self.best_score, optimized_params))
        return self.optimized_params.copy()


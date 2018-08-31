#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides feature tranform and forked from
Created on Thu July 20 2018

@author: cttsai
"""
import gc; gc.enable()

import numpy as np
import pandas as pd

from LibConfigs import logger, file_dir_path
from DataFileIO import DataFileIO


class FeatureImportance(object):
    def __init__(self, default_result_dir=file_dir_path['data']):
        """
        preds = {
            'train_oof' : oof_preds_df,
            'test_oof'  : sub_preds_df,
            'test_full' : test_preds_full,
            'feature_importance': feature_importance_df
        }
        """
        self.result_dir = default_result_dir
        self.importance_series  = pd.Series()
        self.keys = {'importance': 'importance',
                     'feature'   : 'feature'}

    def _analyzeFeatures(self, df):
        if not df.empty:
            self.importance_series = df.groupby(self.keys['feature']).sum()[self.keys['importance']]
            logger.info('feature distribution\n{}'.format(
                    self.importance_series.describe([0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9])))

    def LoadResult(self, result_files):
        if not result_files:
            logger.warning('no result file to rank features')
            return False

        elif len(result_files) == 1:
            ret = DataFileIO().loadHDF('{loc}/{filename}'.format(loc=self.result_dir,
                                    filename=result_files[0]))
            df = ret.get('feature_importance', pd.DataFrame())

        else:
            logger.info('concate {} results to rank features'.format(len(result_files)))
            rets = list()
            for f in result_files:
                rets.append(DataFileIO().loadHDF('{loc}/{filename}'.format(loc=self.result_dir, filename=f)))

            rets = [ret.get('feature_importance', pd.DataFrame()) for ret in rets]
            df = pd.concat(rets, axis=1)

        self._analyzeFeatures(df)

    def GetBlacklist(self, threshold=10.):

        if self.importance_series.empty:
            logger.warning('no feature')
            return list()

        logger.info('create blacklist on score <= {}'.format(threshold))
        ret = self.importance_series.loc[self.importance_series  <= threshold].index.tolist()
        logger.info('return blacklist of {} from {} features'.format(len(ret), len(self.importance_series)))
        return ret

    def CullFeatures(self, x, blacklist=list()):

        if not blacklist:
            logger.warning('empty blacklist')
            return x

        before = x.shape
        x = x[[f for f in x.columns if f not in blacklist]]
        logger.info('shrink from {} to {} by {}'.format(before, x.shape, len(blacklist)))
        return x


def main():
    obj = FeatureImportance('../data/')
    obj.LoadResult(filename='probs_selected_features_level[0]_LightGBM_features[0706]_score[0.783694]_fold[5]_2018-08-25-09-45.hdf5')
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()

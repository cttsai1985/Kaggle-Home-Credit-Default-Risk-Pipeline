#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code provides data and converts to base features and forked from
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
some data processing copied from the Dromosys kernel:
https://www.kaggle.com/dromosys/fork-of-fork-lightgbm-with-simple-features-cee847
Created on Thu Jun 28 2018

@author: cttsai
"""
import gc; gc.enable()

import numpy as np
import pandas as pd

from Utility import InitializeConfigs
from Utility import IdentifyCategoricalColumn, AnyEmptyDataframe, Cast64To32
from LibConfigs import logger, data_provider_refresh_configs

from FeatureTransformer import process_one_hot_encode, process_factorize
from FeatureTransformer import process_interaction, process_deep_interactions
from FeatureTransformer import process_aggregate
from FeatureTransformer import process_decomposition
from FeatureTransformer import process_replace, process_drop_rows

from DataFileIO import DataFileIO


class DataProvider(object):
    """
    **data:
    transformed:
    main: train-test pair
    """
    def __init__(self, IOConfigs={}):

        self.input_path       = IOConfigs.get('input', '../../input')
        self.cache_path       = IOConfigs.get('data', '../data')

        self.data_io_manager  = DataFileIO()
        self.provider_configs = data_provider_refresh_configs

        self.target_column    = 'TARGET'
        self.data_index       = {}
        self.data_raw         = {}
        self.data_processed   = {}
        self.xy_train_test    = {}

        self.cols_categorical = {}
        self.cols_one_hot     = {}

    def _aggregate_pipeline(self, df, cat_cols, configs):
        ret = list()

        for c in configs.get('aggregations', []):

            groupby_cols = c.get('groupby', [])
            if not groupby_cols:
                logger.info("No columns to Aggregate on {}".format(groupby_cols))
                continue

            configs_subset = c.get('subset', {})
            if configs_subset:
                cond_k = configs_subset.get('column_name', 'foobar')
                cond_i = configs_subset.get('conditions', [])

                if cond_k in df.columns and cond_i:
                    sub_df = df.loc[df[cond_k].isin(cond_i)]
                    logger.info("Condictional Aggregate on {}, {}, shape={}".format(cond_k, groupby_cols, sub_df.shape))
                    ret.append(process_aggregate(sub_df, process_configs=c, groupby_cols=groupby_cols, cat_cols=[]))
            else:
                logger.info("Specific Aggregate on {}".format(groupby_cols))
                ret.append(process_aggregate(df, process_configs=c, groupby_cols=groupby_cols, cat_cols=cat_cols))

        ret = [r for r in ret if not r.empty]
        inds = sorted(list(set([r.index.name for r in ret])))
        ret = {ind: pd.concat([r for r in ret if r.index.name == ind], axis=1, join='inner') for ind in inds}

        for k, v in ret.items():
            logger.info("Result Aggregate on {}: {}".format(k, v.shape))

        return ret

    @staticmethod
    def _split_configs(c, name):
        ret = dict()
        for k, v in c.items():
            if 'aggregations' in k:
                ret[k] = [f for f in v if f.get('data', None) == name]
        logger.info('split configs: {}'.format(c))
        return ret

    # Preprocess application_train.csv and application_test.csv
    def _application_train_test(self, configs):
        nan_as_category = configs.get('nan_as_category', False)

        # Read data and merge
        major_index = self.data_index['application_train']
        df      = self.data_raw['application_train']
        test_df = self.data_raw['application_test']
        logger.info("Train samples: {}, test samples: {}".format(df.shape, test_df.shape))
        df = df.append(test_df, sort=False, ignore_index=True)

        df = process_drop_rows(df, process_configs=configs['filter_rows'])
        df = process_factorize(df, process_configs=configs['factorize_columns'])

        if configs.get('onehot_encoding', False):
            df, cat_cols, new_cols = process_one_hot_encode(df, configs['onehot_columns'], nan_as_category)
            self.cols_one_hot.update({'application': new_cols})
        else:
            cat_cols = IdentifyCategoricalColumn(df)

        df = process_replace(df, process_configs=configs['replace_rows'])
        df, interact_cols = process_interaction(df, process_configs=configs['interaction_columns'])

        if configs.get('deep_interactions', []):
            deep_interactions = configs.get('deep_interactions', [])
            for c in deep_interactions:
                df = process_deep_interactions(df, c)

        logger.info('prepare decompostion, application={}'.format(df.shape))
        df_ext = [process_decomposition(df, c) for c in configs['decomposition']]
        df = pd.concat([df] + df_ext, axis=1, join='inner')
        logger.info('finished decompositions, application={}'.format(df.shape))
        df = Cast64To32(df)

        # seperate train test
        # Divide in training/validation and test data
        train_df = df.loc[df[self.target_column].notnull()].reset_index().set_index(major_index)
        test_df  = df.loc[df[self.target_column].isnull()].reset_index().set_index(major_index)
        logger.info("Split into train samples: {}, test samples: {}".format(train_df.shape, test_df.shape))
        del df; gc.collect()

        return train_df, test_df

    # Preprocess bureau.csv and bureau_balance.csv
    def _bureau_and_balance(self, configs):
        current_index = self.data_index['bureau']
        major_index = self.data_index['application_train']
        nan_as_category = configs.get('nan_as_category', False)

        # Read data and merge
        df = self.data_raw['bureau']
        bb = self.data_raw['bureau_balance']
        logger.info("Bureau: {}, Bureau Balance: {}".format(df.shape, bb.shape))

        if configs.get('onehot_encoding', False):
            df, cat_cols, new_cols = process_one_hot_encode(df, configs['onehot_columns'], nan_as_category)
            bb, cat_cols_bb, new_cols_bb = process_one_hot_encode(bb, configs['onehot_columns'], nan_as_category)
            self.cols_one_hot.update({'bureau': new_cols + new_cols_bb})

        agg_configs = self._split_configs(configs.copy(), 'bureau_balance')
        bb_agg = self._aggregate_pipeline(bb, cat_cols_bb, agg_configs)[current_index]
        df = df.set_index(current_index).join(bb_agg, how='left')
        bureau_cat_cols = cat_cols + [c for c in bb_agg if any([True if cc in c else False for cc in cat_cols_bb])]
        #condictional aggregation
        # Bureau: Active credits - using only numerical aggregations
        # Bureau: Closed credits - using only numerical aggregations
        agg_configs = self._split_configs(configs.copy(), 'bureau')
        bureau_agg = self._aggregate_pipeline(df, bureau_cat_cols, agg_configs)[major_index]
        return Cast64To32(bureau_agg)

    # Preprocess previous_applications.csv
    def _previous_application(self, configs):
        current_index = self.data_index['previous_application']
        major_index = self.data_index['application_train']
        nan_as_category = configs.get('nan_as_category', False)

        df = self.data_raw['previous_application']
        logger.info("Previous application: {}".format(df.shape))

        if configs.get('onehot_encoding', False):
            df, cat_cols, new_cols = process_one_hot_encode(df, configs['onehot_columns'], nan_as_category)
            self.cols_one_hot.update({'previous_application': new_cols})
        else:
            cat_cols = IdentifyCategoricalColumn(df)

        df = process_replace(df, process_configs=configs['replace_rows'])
        df, interact_cols = process_interaction(df, process_configs=configs['interaction_columns'])
        # Previous applications categorical features
        # Previous Applications: Approved Applications - only numerical features
        # Previous Applications: Refused Applications - only numerical features
        prev_agg = self._aggregate_pipeline(df, cat_cols, configs)[major_index]

        return Cast64To32(prev_agg)

    # Preprocess POS_CASH_balance.csv
    def _pos_cash_balance(self, configs):
        current_index = self.data_index['pos_cash_balance']
        major_index = self.data_index['application_train']
        nan_as_category = configs.get('nan_as_category', False)

        df = self.data_raw['pos_cash_balance']
        logger.info("pos_cash: {}".format(df.shape))

        if configs.get('onehot_encoding', False):
            df, cat_cols, new_cols = process_one_hot_encode(df, configs['onehot_columns'], nan_as_category)
            self.cols_one_hot.update({'pos_cash': new_cols})
        else:
            cat_cols = IdentifyCategoricalColumn(df)

        pos_cash_agg = self._aggregate_pipeline(df, cat_cols, configs)[major_index]
        return Cast64To32(pos_cash_agg)

    # Preprocess installments_payments.csv
    def _installments_payments(self, configs):
        current_index = self.data_index['installments_payments']
        major_index = self.data_index['application_train']
        nan_as_category = configs.get('nan_as_category', False)

        df = self.data_raw['installments_payments']
        logger.info("installments_payments: {}".format(df.shape))

        cat_cols = []
        if configs.get('onehot_encoding', False):
            df, cat_cols, new_cols = process_one_hot_encode(df, cat_cols, nan_as_category)
            self.cols_one_hot.update({'installments_payments': new_cols})
        else:
            cat_cols = IdentifyCategoricalColumn(df)

        df, interact_cols = process_interaction(df, process_configs=configs['interaction_columns'])
        installments_agg = self._aggregate_pipeline(df, cat_cols, configs)[major_index]
        return Cast64To32(installments_agg)

    # Preprocess credit_card_balance.csv
    def _credit_card_balance(self, configs):
        current_index = self.data_index['credit_card_balance']
        major_index = self.data_index['application_train']
        nan_as_category = configs.get('nan_as_category', False)

        df = self.data_raw['credit_card_balance']
        logger.info("credit_card_balance: {}".format(df.shape))

        cat_cols = []
        if configs.get('onehot_encoding', False):
            df, cat_cols, new_cols = process_one_hot_encode(df, cat_cols, nan_as_category)
            self.cols_one_hot.update({'credit_card_balance' : new_cols})
#        else:
#            cat_cols = IdentifyCategoricalColumn(df)

        credit_card_agg = self._aggregate_pipeline(df, cat_cols, configs)[major_index]
        return Cast64To32(credit_card_agg)

    # Data Input/Output Begin
    def ReadDataCSV(self, configs):
        """
        configs={'application_train'     : {'name' : 'application_train.csv', 'index': 'SK_ID_CURR',},
        """
        data_dict = {k: '{}/{}'.format(self.input_path, data.get('name', None)) for k, data in configs.items()}
        self.data_raw = self.data_io_manager.loadCSV(data_dict)
        self.data_index = {k: data.get('index', None) for k, data in configs.items()}
        return self.data_raw, self.data_index

    def ReadRawHDF(self, configs, filename, limited_by_configs=False):
        """
        configs={'application_train'     : {'name' : 'application_train.csv', 'index': 'SK_ID_CURR',},
        """
        data_dict = {k: None for k, data in configs.items()}
        self.data_raw = self.data_io_manager.loadHDF(filename,
                                                     data_dict,
                                                     limited_by_configs=limited_by_configs)

        self.data_raw = {k: Cast64To32(v) for k, v in self.data_raw.items()}
        self.data_index = {k: data.get('index', None) for k, data in configs.items()}
        return self.data_raw, self.data_index

    def ReadProcessedHDF(self, configs, filename):
        """
        configs={'application_train'     : {'name' : 'application_train.csv', 'index': 'SK_ID_CURR',},
        """
        self.data_processed = self.data_io_manager.loadHDF(filename, {}, limited_by_configs=False)
        self.data_index = {k: data.get('index', None) for k, data in configs.items()}
        return self.data_processed, self.data_index

    def ReadTrainTestHDF(self, configs, filename):
        """
        configs={'application_train'     : {'name' : 'application_train.csv', 'index': 'SK_ID_CURR',},
        """
        self.xy_train_test = self.data_io_manager.loadHDF(filename, configs, limited_by_configs=False)
        return self.xy_train_test

    def SaveFileHDF(self, filename, data, opt_overwrite=True):
        self.data_io_manager.saveHDF(filename, data, opt_overwrite)
    # Data Input/Output --End--

    @staticmethod
    def ReturnTrainTest(configs):
        df_names = ['train_x', 'train_y', 'test_x', 'test_y']
        configs.update({k: pd.DataFrame() for k, v in configs.items() if k not in df_names})
        for df_name in [k for k, v in configs.items() if v.empty]:
            logger.warning("no key as {}".format(df_name))
        # return train_x, train_y, test_x, test_y
        return configs['train_x'], configs['train_y'], configs['test_x'], configs['test_y']

    def CreateTrainTestData(self, configs):
        """
        concat all dataframes to create train and test dataframe
        configs={'application_train' : df},
        """
        train = configs.get('application_train', pd.DataFrame())
        test  = configs.get('application_test',  pd.DataFrame())

        if train.empty or test.empty:
            logger.error('no train and test dataframe')

        excluded = ['application_train', 'application_test']
        for k, v in configs.items():
            if k not in excluded:
                train = train.join(v, how='left')
                test  = test.join(v, how='left')
                logger.info("to_join={}, {}: train={}, test{}".format(k, v.shape, train.shape, test.shape))
                gc.collect()

        # sorted for further
        cols  = sorted(train.columns.tolist())
        train = train[cols]
        test  = test[cols]

        #all process complete
        cols = sorted([f for f in train.columns if f != self.target_column and f in test.columns])
        self.xy_train_test = {
            'train_x': train[cols],
            'train_y': train[self.target_column],
            'test_x': test[cols],
            'test_y': test[self.target_column]}

        del train, test; gc.collect()
        return self.ReturnTrainTest(self.xy_train_test)

    def LoadData(self, data_configs, source='from_csv', prefix='sample'):
        """
        """
        #initialize, reading in configs for data provider itself
        configs_table = pd.DataFrame(self.provider_configs).T
        configs_table['level'] = configs_table['level'].astype(int)
        configs_table.set_index('level', inplace=True)
        configs_table['filename'] = configs_table['filename'].apply(lambda x: x.format(header=prefix) if isinstance(x, str) else None)

        provider_configs = self.provider_configs.get(source, 'from_csv').copy()  #
        refresh_level = provider_configs.get('level')

        # load data at its refresh level
        filename = '{}/{}'.format(self.cache_path, configs_table.loc[refresh_level, 'filename'])
        if refresh_level == 3:
            logger.info("Load Train and Test from Cache")
            self.ReadTrainTestHDF(data_configs['input'], filename)
            if not AnyEmptyDataframe(self.xy_train_test):
                return self.ReturnTrainTest(self.xy_train_test)
            else:
                refresh_level = 2
                logger.warning('No train_test cache to load. Try to refresh at level {}'.format(refresh_level))
                filename = '{}/{}'.format(self.cache_path, configs_table.loc[refresh_level, 'filename'])

        if refresh_level == 2:
            logger.info("Recreate Train and Test")
            self.ReadProcessedHDF(data_configs['input'], filename)
            if AnyEmptyDataframe(self.data_processed):
                refresh_level = 1
                logger.warning('no processed cache to load from disk. Attempt to refresh at level {}'.format(refresh_level))
                filename = '{}/{}'.format(self.cache_path, configs_table.loc[refresh_level, 'filename'])

        if refresh_level == 1:
            logger.info("Process DataFrames from HDF Cashe")
            self.ReadRawHDF(data_configs['input'], filename, limited_by_configs=True)
            if AnyEmptyDataframe(self.data_raw):
                refresh_level = 0
                logger.warning('No raw cache to load. Try to refresh at level {}'.format(refresh_level))

        if refresh_level == 0:
            logger.info("Process DataFrames from CSV")
            self.ReadDataCSV(data_configs['input'])
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[1, 'filename'])
            self.SaveFileHDF(filename, self.data_raw, opt_overwrite=True)

        # process data
        if refresh_level <= 1:
            logger.info("Process DataFrames")
            train_test = self._application_train_test(data_configs['application'])
            self.data_processed = {'application_train': train_test[0],
                                   'application_test' : train_test[1],}

            self.data_processed.update({
                'bureau'               : self._bureau_and_balance(data_configs['bureau']),
                'previous_application' : self._previous_application(data_configs['previous_application']),
                'pos_cash'             : self._pos_cash_balance(data_configs['pos_cash']),
                'credit_card_balance'  : self._credit_card_balance(data_configs['credit_card_balance']),
                'installments_payments': self._installments_payments(data_configs['installments_payments']),
                })

            # save processed
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[2, 'filename'])
            self.SaveFileHDF(filename, self.data_processed, opt_overwrite=True)

        # create train and test
        if refresh_level <= 2:
            self.CreateTrainTestData(self.data_processed)
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[3, 'filename'])
            self.SaveFileHDF(filename, self.xy_train_test, opt_overwrite=True)

        return self.ReturnTrainTest(self.xy_train_test)


def main(argc, argv):

    DataConfigs = InitializeConfigs('../configs/SampleDataConfigs.py').DataConfigs

    dp = DataProvider()
    #dp.ReadRawHDF(DataConfigs, filename='../data/cache_sample_raw.hdf5', limited_by_configs=False)
    #import pdb; pdb.set_trace()

    #dp.LoadData(DataConfigs, source='from_csv', prefix='sample')
    d = dp.LoadData(DataConfigs, source='from_raw_cache', prefix='sample')


    #d = dp.LoadData(DataConfigs, source='from_processed', prefix='sample')
    #d = dp.LoadData(DataConfigs, source='from_train_test', prefix='sample')

    import pdb; pdb.set_trace()

    train_x, train_y = d[0], d[1]

    logger.info('P/N ratio:\n{}'.format(train_y.value_counts(normalize=True).sort_index()))

    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('LossGuideXGB')
    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('DepthWiseXGB')
    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('LinearXGB')
    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('LGBM')
    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('DartLGBM')
    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('BayesianCatBoost')
    #ModelConfigs = InitializeConfigs('../configs/SampleModelConfigs.py').ModelConfigs.get('BernoulliCatBoost')

    #import pdb; pdb.set_trace()
    #from DataModeler import DataModeler
    #eval = DataModeler(ModelConfigs)
    #eval.setupValidation(train_x.iloc[:5000], train_y.iloc[:5000])
    #eval.trainModels(train_x.iloc[:5000], train_y.iloc[:5000])

    #HPOConfigs = ModelConfigs.get("hyperparameter_optimization")
    #from ScikitOptimize import ScikitOptimize
    #from xgboost import XGBClassifier
    #from lightgbm import LGBMClassifier
    #from catboost import CatBoostClassifier
    #hyperparameter_optimize = ScikitOptimize(XGBClassifier, HPOConfigs, task_name='LossGuideXGB')
    #hyperparameter_optimize = ScikitOptimize(XGBClassifier, HPOConfigs, task_name='DepthWiseXGB')
    #hyperparameter_optimize = ScikitOptimize(LGBMClassifier, HPOConfigs, task_name='DartLGBM')
    #hyperparameter_optimize = ScikitOptimize(CatBoostClassifier, HPOConfigs, task_name='LGBM')
    #hyperparameter_optimize.search(train_x.iloc[:10000], train_y.iloc[:10000])

    #hyperparameter_optimize.search(train_x, train_y)

if __name__ == '__main__':
    import sys
    main(len(sys.argv), sys.argv)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provide a class to read and save files
Created on Sat July 21 2018

@author: cttsai
"""
import pandas as pd

from Utility import CheckFileExist
from LibConfigs import logger, hdf5_compress_option, fast_hdf5_compress_option


class DataFileIO(object):
    """
    """
    def __init__(self):
        self.data_lastet_load = {}

    def getLastestLoaded(self):
        return self.data_lastet_load.copy()

    @staticmethod
    def checkFile(filename):
        return CheckFileExist(filename, silent=False)

    @staticmethod
    def loadEmpty(configs):
        return {k: pd.DataFrame() for k in configs.keys()}

    @staticmethod
    def readHDF(filename, configs={}, opt_load=True):
        with pd.HDFStore(filename, 'r', **hdf5_compress_option) as store:
            logger.info("{} contained {} items".format(filename, len(store.keys())))
            for k in store.keys():
                logger.info("{}: {}".format(k, store[k].shape))

            if opt_load and configs:  # load and limited by configs
                ret = {k: pd.DataFrame() for k in configs.keys()}
                ret.update({k.strip('/'): store[k] for k in store.keys() if k.strip('/') in configs.keys()})
                return ret

            if opt_load: # load all saved dataframes
                return {k.strip('/'): store[k] for k in store.keys()}

        return {}

    def showHDF(self, filename):
        self.checkFile(filename)
        self.readHDF(filename, opt_load=False)

    def loadCSV(self, configs={}):
        """
        configs = {'name': 'file_path'}
        return load_data = {'name': dataframe}
        """
        logger.info("Read Data from CSV")
        load_data = {}

        for k, f_path in configs.items():
            if not self.checkFile(f_path):
                continue

            load_data[k] = pd.read_csv(f_path)
            logger.info("Read in {}: from {}, shape={}".format(k, f_path, load_data[k].shape))

        self.data_lastet_load = load_data.copy()
        return load_data

    def loadHDF(self, filename, configs={}, limited_by_configs=True):
        """
        """
        logger.info("Read Data from HDFS")

        if not self.checkFile(filename):
            return self.loadEmpty(configs)

        if limited_by_configs:
            logger.info("Load selected DataFrame Only")
            load_data = self.readHDF(filename, configs, opt_load=True)
        else: # full loaded
            load_data = self.readHDF(filename, opt_load=True)

        for k, v in load_data.items():
            if isinstance(v, pd.DataFrame):
                logger.info('memory usage on {} is {:.3f} MB'.format(k, v.memory_usage().sum() / 1024. ** 2))
        self.data_lastet_load = load_data#.copy()
        return load_data

    def saveHDF(self, filename, data, opt_overwrite=True, opt_fast=False):
        if self.checkFile(filename):
            if not opt_overwrite:
                logger.warning("overwrite is not allowed")
                return False

        compress_option = hdf5_compress_option
        if opt_fast:
            logger.info("use faster compression option")
            compress_option = fast_hdf5_compress_option
        with pd.HDFStore(filename, 'w', **compress_option) as store:
            logger.info("Save to {}".format(filename))
            for k, d in data.items():
                store.put(k, d, format='table')
                #store.put(k, d, format='fixed')
                logger.info("Save {}: {}".format(k, d.shape))

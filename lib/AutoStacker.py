
from itertools import combinations
import pandas as pd
import numpy as np

from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import roc_auc_score

from lib.DataFileIO import DataFileIO
from lib.LibConfigs import logger, file_dir_path
from lib.LibConfigs import filename_submit_mlxtend_meta, filename_submit_mlxtend_base
from lib.LibConfigs import filename_hpo_external
from lib.LibConfigs import filename_mlxtend_meta_features
from lib.LibConfigs import filename_mlxtend_meta_features_external, filename_mlxtend_stacker_external
from lib.Utility import ComposeResultName, CheckFileExist, LoadPickle, SwitchDevice

#base-level, 2nd-level (feature with meta, meta), 3rd-level (meta)
class AutoStacker(object):
    def __init__(self, configs, enable_gpu=False, data_prefix=None):

        self.prefix = data_prefix
        self.input_loc  = file_dir_path['data']
        self.output_loc = file_dir_path['output']
        self.params_loc = file_dir_path['params']
        self.meta_stacker_configs = configs['stacker']
        self.meta_feature_configs = configs['feature']
        self.data_io_manager = DataFileIO()

        self.enable_gpu = enable_gpu

        # mlxtend stacked model
        #self.clf_names = list()
        self.clfs      = list()
        self.clfs_info  = list()

        # meta models from mlxtend stacked model
        self.meta_clfs      = list()
        #self.meta_clf_names = list()
        self.meta_clfs_info = list()

        # store of meta features, meta=collected, agg=concated
        self.X_meta = list()
        self.test_X_meta = list()
        self.X_agg = pd.DataFrame()
        self.test_X_agg = pd.DataFrame()

    @staticmethod
    def _set_submit_filename(level=1, name=None, feature_num=None, score=None, nr_fold=None, seed=None):
        return {
            'level': level,
            'model': name,
            'feature_num': feature_num,
            'score': score,
            'fold': nr_fold,
            'seed': seed,
        }

    def saveSubmit(self, file_stem, preds, template):
        stem = ComposeResultName(file_stem)
        filename = template.format(loc=self.output_loc,
                                   prefix=self.prefix,
                                   stem=stem)
        logger.info('Save predictions to {}'.format(filename))
        preds.to_csv(filename)

    def saveMetaFeatures(self, file_stem, data, stacker_level=False):
        stem = ComposeResultName(file_stem)
        filename = filename_mlxtend_meta_features.format(loc=self.input_loc,
                                                         prefix=self.prefix,
                                                         stem=stem)

        logger.info('Save meta features to {}'.format(filename))
        self.data_io_manager.saveHDF(filename,
                                     data,
                                     opt_overwrite=True,
                                     opt_fast=False)

        if stacker_level:
            filename = filename_mlxtend_stacker_external.format(loc=self.input_loc,
                                                            prefix=self.prefix)
        else:
            filename = filename_mlxtend_meta_features_external.format(loc=self.input_loc, 
                                                                      prefix=self.prefix)
            
        logger.info('export meta features to {}'.format(filename))
        self.data_io_manager.saveHDF(filename,
                                     data,
                                     opt_overwrite=True,
                                     opt_fast=False)

    def loadExternalMeta(self, configs):
        """
        preds = {
            'train_oof' : oof_preds_df,
            'test_oof'  : sub_preds_df,
            'test_full' : test_preds_full,
            'feature_importance': feature_importance_df
        }
        """

        self.X_meta = list()
        self.test_X_meta = list()

        def func(x, by, name):
            return x.groupby(by)['PROBA'].mean().rank(pct=True).rename(name)

        for k, v in configs.items():
            Xs, test_Xs = list(), list()
            for f in v:
                ret = self.data_io_manager.loadHDF('{loc}/{filename}'.format(loc=self.input_loc, filename=f))

                if not ret:
                    continue

                Xs.append(ret.get('train_oof', pd.DataFrame()))
                test_Xs.append(ret.get('test_oof', pd.DataFrame()))

            X      = pd.concat(Xs, axis=0)
            test_X = pd.concat(test_Xs, axis=0)

            X      = func(X.reset_index(), X.index.name, k)
            test_X = func(test_X.reset_index(), test_X.index.name, k)

            self.X_meta.append(X)
            self.test_X_meta.append(test_X)

        filename = filename_mlxtend_meta_features_external.format(loc=self.input_loc, prefix=self.prefix)
        ret = self.data_io_manager.loadHDF(filename)
        if ret:
            df = ret.get('train_meta', pd.DataFrame()).apply(lambda x: x.rank(pct=True))
            self.X_meta.append(df)
            df = ret.get('test_meta', pd.DataFrame()).apply(lambda x: x.rank(pct=True))
            self.test_X_meta.append(df)

        self.X_meta = pd.concat(self.X_meta, axis=1)
        self.test_X_meta = pd.concat(self.test_X_meta, axis=1)
        logger.info('Load Meta {}, {}'.format(self.X_meta .shape, self.test_X_meta.shape))
        return self.X_meta, self.test_X_meta

    def buildMetaFeatures(self, model_zoo):
        for clf in self.meta_feature_configs:
            name                      = clf.get('name', 'foobar')
            use_features_in_secondary = clf.get('use_features', True)
            stratify                  = clf.get('stratify', True)
            nr_folds                  = clf.get('cv', 3)
            seed                      = clf.get('seed', 42)

            bases = [model_zoo.get(c) for c in clf['sources']]
            base_classifiers = [self._create_model_object(clf['model'],
                                                          clf.get('params', dict()),
                                                          clf.get('task', None),
                                                          model_zoo) for clf in bases]

            logger.info('create meta feature extractor')
            self.clfs.append(StackingCVClassifier(base_classifiers,
                     self._create_model_object(clf['meta_classifier'],
                                               clf.get('params', dict()),
                                               clf.get('task', None),
                                               model_zoo),
                     use_probas=True,
                     cv=nr_folds,
                     use_features_in_secondary=use_features_in_secondary,
                     stratify=stratify,
                     store_train_meta_features=True,
                     use_clones=True)
            )
            self.clfs_info.append(self._set_submit_filename(level=1,
                                                            name=name,
                                                            feature_num=None,
                                                            score=None,
                                                            nr_fold=nr_folds,
                                                            seed=seed)
            )
            logger.info('Read in on {} base learners for {}'.format(len(bases), name))

        logger.info('Read in {} meta feature extractors'.format(len(self.clfs)))

    def buildMetaClassifiers(self, model_zoo):
        for clf in self.meta_stacker_configs:
            name                      = clf.get('name', 'foobar')
            use_features_in_secondary = clf.get('use_features', True)
            stratify                  = clf.get('stratify', True)
            nr_folds                  = clf.get('cv', 3)
            seed                      = clf.get('seed', 42)

            bases = clf['base_classifiers']
            logger.info('Learn on {} base learner'.format(len(bases)))
            base_classifiers = [self._create_model_object(clf['model'],
                                                          clf.get('params', dict()),
                                                          clf.get('task', None),
                                                          model_zoo) for clf in bases]
            self.meta_clfs.append(StackingCVClassifier(base_classifiers,
                                  self._create_model_object(clf['meta_classifier'],
                                                            clf.get('params', dict()),
                                                            clf.get('task', None),
                                                            model_zoo),
                                  use_probas=True,
                                  cv=nr_folds,
                                  use_features_in_secondary=use_features_in_secondary,
                                  stratify=stratify,
                                  store_train_meta_features=True,
                                  use_clones=True)
            )
            self.meta_clfs_info.append(self._set_submit_filename(level=2,
                                                                 name=name,
                                                                 feature_num=None,
                                                                 score=None,
                                                                 nr_fold=nr_folds,
                                                                 seed=seed)
            )
            logger.info('Read in on {} base learners for {}'.format(len(bases), name))

        logger.info('Read in {} meta stackers'.format(len(self.meta_clfs)))

    def fitSingleTask(self, clf, X, y, test_X, info={}, nr_class=2, opt_submit=True):
        clf.fit(X.values, y.values)
        X_new = clf.train_meta_features_
        p = pd.DataFrame({'TARGET': clf.predict_proba(X.values)[:, -1]},
                          index=X.index)

        test_X_new = clf.predict_meta_features(test_X.values)
        test_p = pd.DataFrame({'TARGET': clf.predict_proba(test_X.values)[:, -1]},
                               index=test_X.index)

        logger.info('X_meta={}, test_X_meta={}'.format(X_new.shape, test_X_new.shape))
        counter   = [i for i in range(1, X_new.shape[1], nr_class)]
        bases_auc = [roc_auc_score(y, X_new[:, i]) for i in counter]
        #bases_p   = [X_new[:, i]      for i in counter]
        #tests_p   = [test_X_new[:, i] for i in counter]
        if opt_submit:
            l = info['level'] - 1
            info.update({'feature_num': X.shape[1]})
            for i, s in zip(counter, bases_auc):
                p = pd.DataFrame({'TARGET': test_X_new[:, i]}, index=test_X.index)
                info.update({'level': l, 'score': s})
                self.saveSubmit(info,
                                p,
                                template=filename_submit_mlxtend_base)

        return X_new, test_X_new, p, test_p, bases_auc

    def fit_transform(self, X, y, test_X, seed=42):

        X = X.apply(lambda x: np.nan_to_num(x))
        test_X = test_X.apply(lambda x: np.nan_to_num(x))

        for i, (clf, info) in enumerate(zip(self.clfs, self.clfs_info), 1):
            name = info['model']
            logger.info('fit meta feature source: {}'.format(name))
            np.random.seed(info.get('seed', seed))

            X_new, test_X_new, p, test_p, scores = self.fitSingleTask(clf, X, y, test_X, info=info.copy())
            info.update({'feature_num':X_new.shape[1], 'score': max(scores)})
            self.saveSubmit(info,
                            test_p,
                            template=filename_submit_mlxtend_meta)

            columns = ['{}_{}'.format(name, j) for j in range(X_new.shape[1])]
            self.X_meta.append(pd.DataFrame(X_new, index=X.index, columns=columns))
            self.test_X_meta.append(pd.DataFrame(test_X_new, index=test_X.index, columns=columns))

        X      = pd.concat(self.X_meta, axis=1)
        test_X = pd.concat(self.test_X_meta, axis=1)
        logger.info('transform meta feature for X={}, test_X={}'.format(X.shape, test_X.shape))
        self.saveMetaFeatures(info, {'train_meta': X, 'test_meta' : test_X})

    def fit_predict(self, X, y, test_X, seed=42):
        for i, (clf, info) in enumerate(zip(self.meta_clfs, self.meta_clfs_info), 1):
            name = info['model']
            logger.info('fitting meta stackers {}'.format(name))
            np.random.seed(info.get('seed', seed))

            X      = self._process_meta_features(self.X_meta, gamma=None).reindex(X.index)
            test_X = self._process_meta_features(self.test_X_meta, gamma=None).reindex(test_X.index)
            logger.info('processed for X_meta: {}, {}'.format(X.shape, test_X.shape))
            X_new, test_X_new, p, test_p, scores = self.fitSingleTask(clf, X, y, test_X, info=info.copy())
            info.update({'feature_num':X_new.shape[1], 'score': max(scores)})
            self.saveSubmit(info,
                            test_p,
                            template=filename_submit_mlxtend_meta)

            self.saveMetaFeatures(info, {'train_meta': X, 'test_meta': test_X}, stacker_level=True)

    @staticmethod
    def _process_meta_features(X, gamma=None):
        for k in combinations(X.columns, 2):
            X['_X_'.join(k)] = X[list(k)].product(axis=1).apply(lambda x: np.sqrt(x))

        #logger.info('x processed: {}'.format(X_agg.shape))
        return X.apply(lambda x: np.nan_to_num(x))

    def set_model(self, m, params):
        params = SwitchDevice(params, enable_gpu=self.enable_gpu)

        availabe_params = m().get_params()
        if any([k not in availabe_params for k in params.keys()]):
            ret = m(**params)
        else:  # need all parameters in get_params() so safe to call set_params()
            ret = m().set_params(**params)

        logger.info('set {}'.format(ret))
        return ret

    def _create_model_object(self, model, parameters, task, model_zoo):
    # TODO: enable GPU assist

        if task in model_zoo.keys():
            parameters = model_zoo[task].get('params', {})
            logger.info('load parameters {} from model zoo: {}'.format(task, parameters))

            hpo_export = model_zoo[task].get('task', None)
            if hpo_export:
                filename = filename_hpo_external.format(loc=self.params_loc,
                                                        prefix=self.prefix,
                                                        task=hpo_export)
                if CheckFileExist(filename):
                    parameters = LoadPickle(filename)
                    logger.info('Update {} from {}'.format(hpo_export, filename))


        if isinstance(parameters.get('base_estimator', None), str):
            n = parameters.get('base_estimator', None)
            if n in model_zoo.keys():
                params = model_zoo[n].get('params', {})
                sub_model = model_zoo[n].get('model', None)
                logger.info('override parameters {} from model zoo: {}'.format(n, params))
                parameters['base_estimator'] = self.set_model(sub_model, params)

        return self.set_model(model, parameters)



"""
this is forked from two excellent kernels:
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
https://www.kaggle.com/ogrellier/lighgbm-with-selected-features
"""
import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

import ModulePaths
from DataFileIO import DataFileIO
from Utility import ComposeResultName, InitializeConfigs
ext = InitializeConfigs(filename="./external/lighgbm-with-selected-features.py")

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, params, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df; gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
        submission_file_name = "submission_with_selected_features_lgbm_stratified.csv"
        identifier = 'stratified_selected_features'
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
        submission_file_name = "submission_with_selected_features_lgbm.csv"
        identifier = 'selected_features'
    # Create arrays and dataframes to store results

    best_iterations = []
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                             label=train_df['TARGET'].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                             label=train_df['TARGET'].iloc[valid_idx],
                             free_raw_data=False, silent=True)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=500,
            verbose_eval=200
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        print('best iteration: {}'.format(clf.best_iteration))
        best_iterations.append(clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid.label, oof_preds[valid_idx])))
        del clf, dtrain, dvalid
        gc.collect()

    ## INSERTED BEGIN
    oof_preds_df = train_df[['SK_ID_CURR', 'TARGET']].set_index('SK_ID_CURR')
    oof_preds_df['PROBA'] = oof_preds
    sub_preds_df = test_df[['SK_ID_CURR']].set_index('SK_ID_CURR')
    sub_preds_df['PROBA'] = sub_preds

    params['n_estimators'] = max(best_iterations)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(train_df[feats], train_df['TARGET'])
    test_preds_full = sub_preds_df.copy()
    test_preds_full['PROBA'] = clf.predict_proba(test_df[feats])[:,1]

    file_stem = {
        'level': 0,
        'model': 'LightGBM',
        'feature_num': test_df.shape[1],
        'score': roc_auc_score(oof_preds_df['TARGET'].values,
                               oof_preds_df['PROBA'].values),
        'fold': num_folds,}
    stem = ComposeResultName(file_stem)

    data = {
        'train': train_df.set_index('SK_ID_CURR'),
        'test' : test_df.set_index('SK_ID_CURR'),
    }
    DataFileIO().saveHDF('./data/data_{}_{}.hdf5'.format(identifier, stem),
                               data,
                               opt_overwrite=True,
                               opt_fast=False)

    preds = {
        'train_oof' : oof_preds_df,
        'test_oof'  : sub_preds_df,
        'test_full' : test_preds_full,
        'feature_importance': feature_importance_df
    }
    DataFileIO().saveHDF('./data/probs_{}_{}.hdf5'.format(identifier, stem),
                               preds,
                               opt_overwrite=True,
                               opt_fast=False)
    ## INSERTED END

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub_df = test_df[['SK_ID_CURR']].copy()
        sub_df['TARGET'] = sub_preds
        sub_df[['SK_ID_CURR', 'TARGET']].to_csv('output/{}'.format(submission_file_name), index= False)
    #display_importances(feature_importance_df)
    return feature_importance_df


def main(debug=False):
    # LightGBM parameters found by Bayesian optimization
    ParamsLGBM = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            #'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60, # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
            'device': 'gpu',
    }

    num_rows = 10000 if debug else None
    df = ext.application_train_test(num_rows)
    with ext.timer("Process bureau and bureau_balance"):
        bureau = ext.bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau; gc.collect()
    with ext.timer("Process previous_applications"):
        prev = ext.previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev; gc.collect()
    with ext.timer("Process POS-CASH balance"):
        pos = ext.pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos; gc.collect()
    with ext.timer("Process installments payments"):
        ins = ext.installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins; gc.collect()
    with ext.timer("Process credit card balance"):
        cc = ext.credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc; gc.collect()
    with ext.timer("Run LightGBM with kfold"):
        print(df.shape)
        df.drop(ext.features_with_no_imp_at_least_twice, axis=1, inplace=True)
        gc.collect()
        print(df.shape)

        feat_importance = kfold_lightgbm(df,
                                         num_folds=5,
                                         params=ParamsLGBM,
                                         stratified=False,
                                         debug=debug)

        feat_importance = kfold_lightgbm(df,
                                         num_folds=5,
                                         params=ParamsLGBM,
                                         stratified=True,
                                         debug=debug)

if __name__ == "__main__":
    submission_file_name = "submission_with_selected_features.csv"
    with ext.timer("Full model run"):
        main()
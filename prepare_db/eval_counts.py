import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, PROJECT_DIR.parents[0])
sys.path.insert(0, str(PROJECT_DIR.parents[1]) + '/hab_ml')
sys.path.insert(0, PROJECT_DIR.parents[1])
sys.path.insert(0, PROJECT_DIR.parents[2])

# Third party imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from validate_exp.v_utils import concordance_correlation_coefficient, smape, \
    kl_divergence
from scipy.spatial import distance
from scipy import stats

# Project level imports
from hab_ml.utils.logger import Logger

COUNTS_CSV = 'master_counts_v3.csv'


def evaluate_classes(df, sample_method_gold_std_col, sample_method_to_test_col,
                     classes, overall=False, average=False):

    logger.info('Classes selected: {}'.format(classes))
    df = df[df['class'].isin(classes)].reset_index(drop=True)

    # dictionary for storing scores
    eval_metrics = defaultdict(list)

    # evaluate over all classes
    for cls in classes:
        # get class data and assign count data
        temp = df.loc[df['class'] == cls]
        cls_smpl_gold_std, cls_smpl_to_test = temp[sample_method_gold_std_col], temp[
            sample_method_to_test_col]

        # run through all evaluation functions
        eval_metrics = evaluate(eval_metrics, y_true=cls_smpl_gold_std,
                                y_pred=cls_smpl_to_test)
        eval_metrics['class'].append(cls)

    if average:
        eval_metrics['class'].append('average')
        for metric in eval_metrics:
            if metric == 'class':
                continue
            eval_metrics[metric].append(np.mean(eval_metrics[metric]))

    if overall:
        eval_metrics['class'].append('overall')
        overall_y_true, overall_y_pred = df[sample_method_gold_std_col], df[
            sample_method_to_test_col]
        eval_metrics = evaluate(eval_metrics, y_true=overall_y_true,
                                y_pred=overall_y_pred)

    return eval_metrics


def get_eval_fn():
    metrics = {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'smape': smape,
        'ccc': concordance_correlation_coefficient,
        'bray curtis': distance.braycurtis,
        'pearson': stats.pearsonr,
        'kl': kl_divergence,
    }
    return metrics


def evaluate(eval_metrics, y_true, y_pred):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # get evaluation functions
    eval_fns = get_eval_fn()

    # run through all evaluation functions
    for metric, eval_fn in eval_fns.items():
        # evaluate and save score
        score = eval_fn(y_true, y_pred)
        if not isinstance(score, float):
            score = score[0]
        logger.debug('{:15} {}'.format(metric, score))
        eval_metrics[metric].append(score)
    return eval_metrics


def evaluate_counts(input_dir, sample_method_gold_std, sample_method_to_test,
                    classes, count='raw count'):
    # Initialize logging
    suffix = f'{sample_method_gold_std.replace(" ", "-")}-{sample_method_to_test.replace(" ", "-")}'
    logger = logging.getLogger(__name__)

    sample_method_gold_std_col = f'{sample_method_gold_std} {count}'
    sample_method_to_test_col = f'{sample_method_to_test} {count}'

    data = pd.read_csv(os.path.join(input_dir, COUNTS_CSV))
    df = data.copy()
    df = df.dropna()

    logger.info('Dataset size: {}'.format(df.shape[0]))
    logger.info('Total dates: {}'.format(df['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(df['class'].nunique()))

    # transform dataset

    # evaluation
    eval_metrics = evaluate_classes(df, sample_method_gold_std_col,
                                    sample_method_to_test_col,
                                    classes, overall=True, average=True)

    # Create dataframe and save to csv
    eval_df = pd.DataFrame(eval_metrics)
    logger.info('error and agreements\n{}'.format('-' * 30))
    results = dict(zip(eval_metrics.keys(),
                       eval_df.loc[eval_df['class'] == 'overall'].values.tolist()[0]))
    for metric, score in results.items():
        if metric == 'class':
            continue
        logger.info(f'{metric}: {score:.02f}')

    logger.info('SMAPE Results\n{}'.format('-' * 30))
    logger.info(eval_df['smape'].describe())

    logger.info('KLDIV Results\n{}'.format('-' * 30))
    logger.info(eval_df['kl'].describe())

    csv_fname = os.path.join(input_dir, 'eval_{}.csv'.format(suffix))
    eval_df.to_csv(csv_fname, index=False)
    logger.info(f'Saved eval csv as {csv_fname}')

    return results['smape'], results['kl']

if __name__ == '__main__':

    classes = ['Akashiwo', 'Ceratium falcatiforme or fusus', 'Ceratium furca',
               'Chattonella', 'Cochlodinium', 'Gyrodinium', 'Lingulodinium polyedra',
               'Prorocentrum micans', 'Pseudo-nitzschia chain']

    # import random
    # for smpl in range(1, len(classes) + 1):
    #     print('{}:{},'.format(smpl, random.sample(classes, smpl)))
    RANDOM_SMPLED_CLSSES = {
    }

    input_dir = '/data6/phytoplankton-db/counts/'
    log_fname = os.path.join(input_dir, 'eval_counts.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')
    logger = logging.getLogger('create-csv')

    SMAPE_VS_CLASS_EXP = False
    COUNT = 'raw count'
    logger.info('Count form: {}'.format(COUNT))

    MICRO = 'micro'
    LAB_GT = 'lab gtruth'
    LAB_PRED = 'lab predicted'
    PIER_GT = 'pier gtruth'
    PIER_PRED = 'pier predicted'

    settings = [(MICRO, LAB_GT),
                (MICRO, LAB_PRED),
                (MICRO, PIER_GT),
                (MICRO, PIER_PRED),
                (LAB_GT, LAB_PRED),
                (PIER_GT, PIER_PRED)]

    for (smpl_gold, smpl_test) in settings:
        Logger.section_break(f'{smpl_gold} vs {smpl_test}')
        evaluate_counts(input_dir,
                        sample_method_gold_std=smpl_gold,
                        sample_method_to_test=smpl_test, classes=classes,
                        count=COUNT)

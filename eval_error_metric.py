"""

evaluate each stat

Usage
-----
>>> from eval_error_metric import compare_metrics
>>> from validate_exp.stat_fns import ERROR_STATS, AGREEMENT_STATS
>>> logger = logging.getLogger(__name__)
>>> logger.setLevel(logging.INFO)

# Load counts dataset
>>> data_ = {}
>>> data_['camera'] = pd.read_csv(COUNTS_CSV['counts'])
>>> data_['classifier'] = pd.read_csv(COUNTS_CSV['counts-v9'])

# error_scores, agreemenet_scores = main()
>>> error_scores = compare_metrics(ERROR_STATS, data=data_)
>>> agreement_scores = compare_metrics(AGREEMENT_STATS, data=data_)
"""
import argparse
import logging
import os

import pandas as pd

from counts_analysis.c_utils import COUNTS_CSV, CLASSES, set_counts, set_settings, \
    CORRELATED_CLASSES
from eval_counts import evaluate_settings, evaluate_counts
from hab_ml.utils.logger import Logger
from validate_exp.stat_fns import *

COMPARE_ALL_FLAG = False
FILTER_CLASSES_FLAG = False
CORRELATED_CLASSES_FLAG = False

def run_comparisons():
    logger = logging.getLogger(__name__)

    # Load counts dataset
    data = {}
    data['camera'] = pd.read_csv(COUNTS_CSV['counts'])
    data['classifier'] = pd.read_csv(COUNTS_CSV['counts-v9'])

    def filter_classes(df, classes):
        return df[df['class'].isin(classes)].reset_index(drop=True)

    if FILTER_CLASSES_FLAG:
        logger.info('FILTER CLASSES: {}'.format(FILTER_CLASSES_FLAG))
        classes = CORRELATED_CLASSES if CORRELATED_CLASSES_FLAG else CLASSES
        logger.info('CORRELATED CLASSES: {}'.format(CORRELATED_CLASSES_FLAG))
        data['camera'] = filter_classes(data['camera'], classes)
        data['classifier'] = filter_classes(data['classifier'], classes)

    data_ = data['camera']
    logger.info('Dataset size: {}'.format(data_.shape[0]))
    logger.info('Total dates: {}'.format(data_['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(data_['class'].nunique()))

    # Initialize stats to compute
    if COMPARE_ALL_FLAG:
        error_stats = ERROR_STATS
        agreement_stats = AGREEMENT_STATS
    else:
        error_stats = [mase]
        agreement_stats = [pearson]

    logger.info(
        f'Error Metric Comparisons to Make\n{[st.__name__ for st in error_stats]}')
    logger.info(
        f'Agreement Metric Comparisons to Make\n{[st.__name__ for st in agreement_stats]}')

    # evaluate each stat
    logger.info('Starting Error Metric Comparisons...')
    error_scores = compare_metrics(error_stats, data=data)

    # evaluate each stat
    logger.info('Starting Agreement Metric Comparisons...')
    agreement_scores = compare_metrics(agreement_stats, data=data)

    Logger.section_break("Overall")
    grp_err = error_scores.groupby("stat")
    grp_aggr = agreement_scores.groupby("stat")
    pd.options.display.float_format = '{:,.2f}'.format
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', -1):
        cameras = ['lab - micro', 'pier - micro', 'pier - lab']
        classifier = ['clssf lab', 'clssf pier']
        Logger.section_break('Avg scores')
        logger.info(f'Camera Counts\n{"-" * 30}')
        logger.info(f'{grp_err[cameras].mean().T}')
        logger.info(f'\n{grp_aggr[cameras].mean().T}\n')

        logger.info(f'Classifier Counts\n{"-" * 30}')
        logger.info(f'{grp_err[classifier].mean().T}')
        logger.info(f'\n{grp_aggr[classifier].mean().T}\n')

        Logger.section_break('Median scores')
        logger.info(f'Camera Counts\n{"-" * 30}')
        logger.info(f'{grp_err[cameras].median().T}')
        logger.info(f'\n{grp_aggr[cameras].median().T}\n')

        logger.info(f'Classifier Counts\n{"-" * 30}')
        logger.info(f'{grp_err[classifier].median().T}')
        logger.info(f'\n{grp_aggr[classifier].median().T}\n')

    return error_scores, agreement_scores


def compare_metrics(stats, data, plot=True, verbose=False):
    """Compare each stat and print out a table of the comparison

    Table for camera, then classifier

    """
    from collections import defaultdict

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Set counts, settings, & clsifier datasets
    rc_counts = set_counts('gtruth', 'raw count', micro_default=True)
    settings = set_settings(rc_counts)

    # === Set classifier gtruth vs predictions
    grouping = 'class'
    data['camera-grp'] = data['camera'].groupby(grouping)
    data['classifier-grp'] = data['classifier'].groupby(grouping)

    stat_dict = {}
    for stat in stats:
        columns = ['class', 'lab - micro', 'pier - micro', 'pier - lab', 'clssf lab',
                   'clssf pier']
        stat_df = pd.DataFrame(columns=columns)
        class_scores = defaultdict(dict)
        Logger.section_break(stat.__name__, log_level=10)

        grouped_data = data['camera-grp']
        for idx, (grp_name, grp_data) in enumerate(grouped_data):
            logger.debug('\n\n\nGrouping: {}\nStat: {}'.format(grp_name, stat.__name__))

            # Evaluate camera counts
            camera_scores = evaluate_settings(settings, stat, grp_data,
                                              do_bootstrap=False)

            # Evaluate classifier counts
            grp_data_ = data['classifier-grp'].get_group(grp_name)
            classifier_scores = evaluate_classifier_counts(stat, grp_data_)
            logger.debug('\n')

            score_data = [grp_name] + camera_scores + classifier_scores
            score_data = dict(zip(columns, score_data))
            stat_df = stat_df.append(score_data, ignore_index=True)
        stat_dict[stat.__name__] = stat_df

    Logger.section_break('Summary')
    main_df = pd.DataFrame()
    for stat, score_df in stat_dict.items():
        score_df = score_df.replace([np.inf, -np.inf], np.nan)
        print_metric(stat, score_df, verbose=verbose)

        score_df['stat'] = stat
        main_df = main_df.append(score_df)
    logger.info('\n')

    cols = list(main_df.columns)
    cols.insert(1, cols.pop())
    main_df = main_df[cols].reset_index(drop=True)
    return main_df


def evaluate_classifier_counts(stat, data):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # === Set classifier gtruth vs predictions
    cameras = ['lab', 'pier']
    logger.debug('#===== Classifier Counts ====#')
    scores = {}
    logger.debug('{:22}    {:22} Score'.format('Pred (Y)', 'Gtruth (X)'))
    logger.debug(f'{"-" * 70}')
    for cam in cameras:
        gtruth, pred = f'{cam} gtruth raw count', f'{cam} predicted raw count'
        n = data[f'{cam} gtruth raw count'].sum() if stat.__name__ == 'mae' else None
        scores[(gtruth, pred)] = evaluate_counts(stat=stat, data=data, gtruth=gtruth,
                                                 pred=pred, n=n)
        logger.debug('{:25} {:25} {}'.format(pred, gtruth, scores[(gtruth, pred)]))
    return list(scores.values())


def print_metric(stat, score_df, verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f'\n\n#======== {stat.upper()} ========#\n{"-" * 40}')
    pd.options.display.float_format = '{:,.2f}'.format
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', -1):
        logger.info(f'\n{score_df}')
        scores = score_df[score_df.columns[1:]].describe()
        if verbose:
            logger.info(scores)
        else:
            logger.info('\nAvg Score\n{}\n{}'.format("-" * 15, scores.loc['mean']))
            logger.info('\nMedian Score\n{}\n{}'.format("-" * 15, scores.loc['50%']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Counts')
    parser.add_argument('--input_dir', type=str,
                        default='/data6/phytoplankton-db/counts/',
                        help='Count data directory to evaluate counts from')
    args = parser.parse_args()

    # Initialize logger
    log_fname = os.path.join(args.input_dir, 'eval_error_metric-correlated_cls_only.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Error/Agreement Comparison')

    run_comparisons()

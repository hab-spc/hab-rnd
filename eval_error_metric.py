"""

evaluate each stat
"""
import argparse
import logging
import os

import pandas as pd

from counts_analysis.c_utils import COUNTS_CSV, CLASSES, set_counts, set_settings
from eval_counts import evaluate_settings, evaluate_counts
from hab_ml.utils.logger import Logger
from validate_exp.stat_fns import *

COMPARE_ALL_FLAG = True
FILTER_CLASSES_FLAG = False


def run_comparisons():
    logger = logging.getLogger(__name__)

    # Load counts dataset
    df = pd.read_csv(COUNTS_CSV['counts'])
    df_ = df[df['class'].isin(CLASSES)].reset_index(drop=True)

    data = df_.copy() if FILTER_CLASSES_FLAG else df.copy()
    logger.info('Dataset size: {}'.format(data.shape[0]))
    logger.info('Total dates: {}'.format(data['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(data['class'].nunique()))

    # Initialize stats to compute
    if COMPARE_ALL_FLAG:
        error_stats = ERROR_STATS
        agreement_stats = AGREEMENT_STATS
    else:
        error_stats = []
        agreement_stats = []

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
    pd.options.display.float_format = '{:,.2f}'.format
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', -1):
        cameras = ['setting 1', 'setting 2', 'setting 3']
        classifier = ['lab', 'pier']
        logger.info(f'Camera Counts\n{"-" * 30}')
        logger.info(f'\n{error_scores.groupby("stat")[cameras].mean().T}')
        logger.info(f'\n{agreement_scores.groupby("stat")[cameras].mean().T}')

        logger.info(f'Classifier Counts\n{"-" * 30}')
        logger.info(f'\n{error_scores.groupby("stat")[classifier].mean().T}')
        logger.info(f'\n{agreement_scores.groupby("stat")[classifier].mean().T}')

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
    grouping = 'datetime'
    grouped_data = data.groupby(grouping)
    stat_dict = {}
    for stat in stats:
        columns = ['class', 'setting 1', 'setting 2', 'setting 3', 'lab', 'pier']
        stat_df = pd.DataFrame(columns=columns)
        class_scores = defaultdict(dict)
        Logger.section_break(stat.__name__, log_level=10)
        for idx, (grp_name, grp_data) in enumerate(grouped_data):
            logger.debug('\n\n\nGrouping: {}\nStat: {}'.format(grp_name, stat.__name__))

            camera_scores = evaluate_settings(settings, stat, grp_data,
                                              do_bootstrap=False)

            classifier_scores = evaluate_classifier_counts(stat, grp_data)
            logger.debug('\n')

            data = [grp_name] + camera_scores + classifier_scores
            data = dict(zip(columns, data))
            stat_df = stat_df.append(data, ignore_index=True)
        stat_dict[stat.__name__] = stat_df

    Logger.section_break('Summary')
    main_df = pd.DataFrame()
    for stat, score_df in stat_dict.items():
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
    logger.setLevel(logging.INFO)

    # === Set classifier gtruth vs predictions
    cameras = ['lab', 'pier']
    logger.debug('#===== Classifier Counts ====#')
    scores = {}
    logger.debug('{:22}    {:22} Score'.format('Pred (Y)', 'Gtruth (X)'))
    logger.debug(f'{"-" * 70}')
    for cam in cameras:
        gtruth, pred = f'{cam} gtruth raw count', f'{cam} predicted raw count'
        scores[(gtruth, pred)] = evaluate_counts(stat=stat, data=data, gtruth=gtruth,
                                                 pred=pred)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Counts')
    parser.add_argument('--input_dir', type=str,
                        default='/data6/phytoplankton-db/counts/',
                        help='Count data directory to evaluate counts from')
    args = parser.parse_args()

    # Initialize logger
    log_fname = os.path.join(args.input_dir, 'eval_error_metric.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Error/Agreement Comparison')

    run_comparisons()

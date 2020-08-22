import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, PROJECT_DIR.parents[0])
sys.path.insert(0, PROJECT_DIR.parents[1])
sys.path.insert(0, PROJECT_DIR.parents[2])

# Third party imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial import distance
from scipy import stats

# Project level imports
from validate_exp.stat_fns import concordance_correlation_coefficient, kl_divergence, \
    smape, mase
from validate_exp.v_utils import set_counts, get_confidence_limit
from counts_analysis.c_utils import CLASSES, set_counts, set_settings, \
    CORRELATED_CLASSES
from hab_ml.utils.logger import Logger

FILTER_CLASSES_FLAG = True
CORRELATED_CLASSES_FLAG = False


def main():
    logger = logging.getLogger('create-csv')

    COUNT = 'cells/mL'
    logger.info('Count form: {}'.format(COUNT))

    # Set count forms
    volumetric_counts = set_counts('gtruth', 'cells/mL', micro_default=True)
    rc_counts = set_counts('gtruth', 'raw count', micro_default=True)
    rc_counts_pred = set_counts('predicted', 'raw count', micro_default=True)
    nrmlzd_counts = set_counts('gtruth', 'nrmlzd raw count', micro_default=True)
    rel_counts = set_counts('gtruth', 'relative abundance', micro_default=False)
    rel_counts = ['micro cells/mL relative abundance'] + list(rel_counts[1:])

    # Set settings
    settings_ = [set_settings(count) for count in [volumetric_counts, rc_counts,
                                                   nrmlzd_counts, rel_counts,
                                                   rc_counts_pred]]
    count_forms = dict(zip(['volumetric', 'raw', 'nrmlzd', 'relative', 'predicted raw'],
                           settings_))

    # Set evaluation metric
    stat = mase

    # Load count dataset
    count_version = 'counts_v10'
    high = '/data6/phytoplankton-db/counts/master_counts_v8.csv'
    df = pd.read_csv(high)
    if count_version.split('_')[1] == 'v10':
        logger.info('V10> detected. Filtering for HAB classes')
        df = df[df['class'] != "Other"].reset_index(drop=True)

    def filter_classes(df, classes):
        return df[df['class'].isin(classes)].reset_index(drop=True)

    if FILTER_CLASSES_FLAG:
        logger.info('FILTER CLASSES: {}'.format(FILTER_CLASSES_FLAG))
        classes = CORRELATED_CLASSES if CORRELATED_CLASSES_FLAG else CLASSES
        logger.info('CORRELATED CLASSES: {}'.format(CORRELATED_CLASSES_FLAG))
        df = filter_classes(df, classes)

    def compute_relative_abundance(raw_count, data):
        if 'micro' in raw_count:
            relative_column = 'micro cells/mL relative abundance'
        else:
            relative_column = f'{raw_count.split()[0]} {raw_count.split()[1]} relative abundance'
        # Compute relative abundance
        data[relative_column] = data.groupby('datetime')[raw_count].apply(
            lambda x: x / x.sum() if sum(x) != 0 else x)
        return data

    # for rc in rc_counts:
    #     df = compute_relative_abundance(rc, df)

    logger.info('Dataset size: {}'.format(df.shape[0]))
    logger.info('Total dates: {}'.format(df['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(df['class'].nunique()))

    # Evaluate count forms
    settings_score = compare_count_forms(count_forms, stat, df)
    Logger.section_break("Overall")
    grp_scores = settings_score.groupby('count form')
    settings = ['lab - micro', 'pier - micro', 'pier - lab']
    pd.options.display.float_format = '{:,.2f}'.format
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', -1):
        logger.info(f'Average Scores\n{"-" * 30}')
        logger.info(f'{grp_scores[settings].mean().T}')

        logger.info(f'Median Scores\n{"-" * 30}')
        logger.info(f'{grp_scores[settings].median().T}')

        logger.info(f'Confidence Limits\n{"-" * 30}')
        for grp_name, grp_df in grp_scores:
            logger.info(grp_name)
            for setting in settings:
                logger.info('{} {:.1f} confidence interval {:.2f}% and {:.2f}%'.
                            format(setting, *get_confidence_limit(grp_df[setting])))

    return settings_score


def compare_count_forms(count_forms, stat, data):
    logger = logging.getLogger(__name__)

    # Define grouping (classes/datetime)
    grouping = 'class'
    logger.info('Grouping: {}'.format(grouping.upper()))
    grouped_data = data.groupby(grouping)

    stat_dict = {}
    for count_form, settings in count_forms.items():

        columns = ['class', 'lab - micro', 'pier - micro', 'pier - lab']
        stat_df = pd.DataFrame(columns=columns)

        for idx, (grp_name, grp_data) in enumerate(grouped_data):

            # Evalute settings for this grouping
            if count_form == 'raw count' and grp_name == 'Ceratium furca':
                print('debugging')

            # Get scores for each setting
            camera_scores = evaluate_settings(settings, stat, grp_data,
                                              do_bootstrap=False)

            # Save scores (class, score set1, score set2, score set3) format
            score_data = dict(zip(columns, [grp_name] + camera_scores))
            stat_df = stat_df.append(score_data, ignore_index=True)
        # Save dataframe for each count form into dictionary
        stat_df = stat_df.replace([np.inf, -np.inf], np.nan)
        stat_dict[count_form] = stat_df

    # Print stat dictionary for each count form (class scores for each setting)
    # then return the stat dictionary as a dataframe
    Logger.section_break('Summary')
    main_df = pd.DataFrame()
    for count_form, score_df in stat_dict.items():
        print_metric(count_form, score_df, verbose=False)

        score_df['count form'] = count_form
        main_df = main_df.append(score_df)
    logger.info('\n')

    cols = list(main_df.columns)
    cols.insert(1, cols.pop())
    main_df = main_df[cols].reset_index(drop=True)
    return main_df


def evaluate_settings(settings, stat, df, classes=None, do_bootstrap=True):
    """

    Evaluate

    Args:
        settings:
        stat:
        df:
        classes:
        do_bootstrap:

    Returns:

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)

    if classes:
        logger.info('Classes selected: {}'.format(classes))
        df = df[df['class'].isin(classes)].reset_index(drop=True)

    scores = {}
    logger.debug('#===== Camera Counts =====#')
    logger.debug('{:22}    {:22} Score'.format('Pred (Y)', 'Gtruth (X)'))
    logger.debug(f'{"-" * 70}')
    for setting, (gtruth, pred) in settings.items():
        scores[(gtruth, pred)] = evaluate_counts(df, gtruth=gtruth, pred=pred, stat=stat)
        logger.debug('{:25} {:25} {}'.format(pred, gtruth, scores[(gtruth, pred)]))

    if do_bootstrap:
        logger.info('Bootstrapping...')
        setting_lbls = ['micro-lab', 'micro-pier', 'lab-pier'] * 2
        score_settings = dict(zip(settings, setting_lbls))
        booted_eval_metrics = {}
        for stat in [smape, concordance_correlation_coefficient]:
            logger.info(stat.__name__)
            for setting, (gtruth, pred) in settings.items():
                booted_eval_metrics[setting] = bootstrap(x=gtruth, y=pred,
                                                         stats=stat, data=df)
            print_eval_bootstrap(**booted_eval_metrics)

    return list(scores.values())


def evaluate_counts(data, gtruth, pred, stat, n=None):
    """ Evaluates a setting and save it for plotting

    Args:
        input_dir (str): Parent directory to read counts from
        sample_method_gold_std (str): Gold standard column name
        sample_method_to_test (str): Tested column name
        classes (list): Classes

    """
    # Initialize logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    x, y = data[gtruth], data[pred]
    score = stat(x, y, n=n) if n else stat(x, y)

    if not isinstance(score, float):
        score = score[0]

    return score


def evaluate_classes(df, gtruth, pred,
                     overall=False, average=False):
    """ Evaluate over all classes

    Args:
        df: Dataset
        gtruth (str): Column name to compare against (gtruth)
        pred (str): Column name to test
        overall (bool):  Flag to compute aggregated score. Default False
        average (bool): Flag to compute average of all class scores. Default False

    Returns:
        dict: Evaluation metrics

    """
    logger = logging.getLogger(__name__)

    # dictionary for storing scores
    eval_metrics = defaultdict(list)

    # evaluate over all classes
    classes = sorted(df['class'].unique())
    for cls in classes:
        # get class data and assign count data
        temp = df.loc[df['class'] == cls]
        cls_smpl_gold_std, cls_smpl_to_test = temp[gtruth], temp[pred]

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
        overall_y_true, overall_y_pred = df[gtruth], df[
            pred]
        eval_metrics = evaluate(eval_metrics, y_true=overall_y_true,
                                y_pred=overall_y_pred)

    return eval_metrics


def get_eval_fn():
    """ Returns consolidated dictionary of evaluation functions to run

    Returns:
        dict: Evaluation metrics
    """
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
    """ Evaluates count data and returns dictionary of scores

    Args:
        eval_metrics (dict): Dictionary to store scores
        y_true (list): Gtruth data points
        y_pred (list): Predicted data points

    Returns:
        dict: scores

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

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


def print_eval(scores):
    logger = logging.getLogger(__name__)
    Logger.section_break('Overall SMAPE Scores')
    for idx, (setting, score) in enumerate(scores.items()):
        logger.info('{:50} SMAPE:{:0.2f}'.format(str(setting), score[0]))


    Logger.section_break('Overall KLDIV Scores')
    for idx, (setting, score) in enumerate(scores.items()):
        logger.info('{:50} KLDIV:{:0.2f}'.format(str(setting), score[1]))


def print_eval_bootstrap(**kwargs):
    logger = logging.getLogger(__name__)

    def print_error(data):
        logger.info(f'COUNT: {len(data)}')
        logger.info(f'AVG: {np.mean(data)}')
        logger.info(f'MEDIAN: {np.median(data)}')
        logger.info(f'STD DEV: {np.std(data)}')
        logger.info(f'VAR: {np.var(data)}')
        logger.info('%.1f confidence interval %.2f%% and %.2f%%\n' % (
            get_confidence_limit(data)))

    for idx, dist_title in enumerate(kwargs):
        logger.info('{}\n{}'.format(dist_title, '-' * 30))
        print_error(kwargs[dist_title])


def bootstrap(x, y, data, stats, n_iterations=10000):
    from sklearn.utils import resample

    score = []
    n_size = int(len(data) * .8)
    bootstrap_size = 0
    for i in range(n_iterations):
        bootstrap_sample = resample(data, n_samples=n_size)
        score.append(stats(bootstrap_sample[x], bootstrap_sample[y]))
        bootstrap_size += len(bootstrap_sample)
    #         if i % 1000 == 0:
    #             print(f'{i}/{n_iterations} completed. Bootstrap sample size: {bootstrap_size}')

    return score


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
            logger.info('\nStd Dev\n{}\n{}'.format("-" * 15, scores.loc['std']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Counts')
    parser.add_argument('--input_dir', type=str,
                        default='/data6/phytoplankton-db/counts/',
                        help='Count data directory to evaluate counts from')
    args = parser.parse_args()

    # Initialize logging
    input_dir = args.input_dir
    log_fname = os.path.join(input_dir, 'eval_counts.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')

    main()

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
from validate_exp.v_utils import concordance_correlation_coefficient, smape, \
    kl_divergence, set_counts, get_confidence_limit
from hab_ml.utils.logger import Logger

COUNTS_CSV = 'master_counts_v7.csv'


def main(args):
    # Excluded 3 classes atm
    classes = ['Akashiwo',
               'Ceratium falcatiforme or fusus',
               'Ceratium furca',
               'Chattonella',
               'Cochlodinium',
               'Lingulodinium polyedra',
               'Prorocentrum micans']

    # classes = ['Akashiwo',
    #  'Ceratium falcatiforme or fusus',
    #  'Ceratium furca',
    #  'Chattonella',
    #  'Cochlodinium',
    #  'Gyrodinium',
    #  'Lingulodinium polyedra',
    #  'Prorocentrum micans',
    #  'Pseudo-nitzschia chain']

    # Initialize logging
    input_dir = args.input_dir
    log_fname = os.path.join(input_dir, 'eval_counts.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')
    logger = logging.getLogger('create-csv')

    COUNT = 'cells/mL'
    logger.info('Count form: {}'.format(COUNT))

    # Load count dataset
    df = pd.read_csv(os.path.join(input_dir, COUNTS_CSV))

    logger.info('Dataset size: {}'.format(df.shape[0]))
    logger.info('Total dates: {}'.format(df['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(df['class'].nunique()))

    # Set count forms
    MICRO_ML, LAB_ML, PIER_ML = set_counts('gtruth', 'cells/mL')
    _, LAB_P_ML, PIER_P_ML = set_counts('predicted', 'cells/mL')
    MICRO_RC, LAB_RC, PIER_RC = set_counts('gtruth', 'raw count')
    MICRO, LAB_P_RC, PIER_P_RC = set_counts('predicted', 'raw count')
    _, LAB_NRC, PIER_NRC = set_counts('gtruth', 'nrmlzd raw count')
    _, LAB_P_NRC, PIER_P_NRC = set_counts('predicted', 'nrmlzd raw count')

    settings = [
        # Vol vs Vol
        (MICRO_ML, LAB_ML),
        (MICRO_ML, PIER_ML),
        (LAB_ML, PIER_ML),

        # Volume vs Raw Count
        (MICRO_ML, LAB_RC),
        (MICRO_ML, PIER_RC),
        (LAB_RC, PIER_RC),

        # Volume vs Normalized Raw Count
        (MICRO_ML, LAB_NRC),
        (MICRO_ML, PIER_NRC),
        (LAB_NRC, PIER_NRC),
    ]

    # Transform dataset
    # df, tf_settings = transform_dataset(df, (MICRO_ML, LAB_RC, PIER_RC, LAB_NRC,
    #                                          PIER_NRC))
    # settings.extend(tf_settings)

    # Evalute counts
    evaluate_settings(settings, df, classes)


def evaluate_settings(settings, df, classes, do_bootstrap=True):
    logger = logging.getLogger(__name__)
    logger.info('Classes selected: {}'.format(classes))
    df = df[df['class'].isin(classes)].reset_index(drop=True)

    scores = {}
    for (smpl_gold, smpl_test) in settings:
        Logger.section_break(f'{smpl_gold} vs {smpl_test}')
        scores[(smpl_gold, smpl_test)] = evaluate_counts(df,
                                                         gtruth_smpl_mthd=smpl_gold,
                                                         exp_smpl_mthd=smpl_test)
    print_eval(scores)

    if do_bootstrap:
        logger.info('Bootstrapping...')
        setting_lbls = ['micro-lab', 'micro-pier', 'lab-pier'] * 2
        score_settings = dict(zip(settings, setting_lbls))
        booted_eval_metrics = {}
        for stat in [smape, concordance_correlation_coefficient]:
            logger.info(stat.__name__)
            for (smpl_gold, smpl_test), setting_lbl in score_settings.items():
                booted_eval_metrics[setting_lbl] = bootstrap(x=smpl_gold, y=smpl_test,
                                                             stats=stat, data=df)
            print_eval_bootstrap(**booted_eval_metrics)

    return scores


def evaluate_counts(data, gtruth_smpl_mthd, exp_smpl_mthd):
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

    # Load dataset
    df = data.copy()
    df = df.dropna()

    logger.debug('Dataset size: {}'.format(df.shape[0]))
    logger.debug('Total dates: {}'.format(df['datetime'].nunique()))
    logger.debug('Total classes: {}\n'.format(df['class'].nunique()))

    # evaluation
    eval_metrics = evaluate_classes(df, gtruth_smpl_mthd,
                                    exp_smpl_mthd, overall=True, average=True)

    # Create dataframe and save to csv
    eval_df = pd.DataFrame(eval_metrics)
    logger.info('error and agreements\n{}'.format('-' * 30))
    results = dict(zip(eval_metrics.keys(),
                       eval_df.loc[eval_df['class'] == 'average'].values.tolist()[0]))
    for metric, score in results.items():
        if metric == 'class':
            continue
        logger.info(f'{metric}: {score:.02f}')

    logger.debug('\nSMAPE Results\n{}'.format('-' * 30))
    logger.debug(eval_df['smape'].describe())

    logger.info('\nCCC Results\n{}'.format('-' * 30))
    logger.info(eval_df['ccc'].describe())
    # no savinng at the moment
    return results['smape'], results['ccc']

def evaluate_classes(df, sample_method_gold_std_col, sample_method_to_test_col,
                     overall=False, average=False):
    """ Evaluate over all classes

    Args:
        df: Dataset
        sample_method_gold_std_col (str): Column name to compare against (gtruth)
        sample_method_to_test_col (str): Column name to test
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Counts')
    parser.add_argument('--input_dir', type=str,
                        default='/data6/phytoplankton-db/counts/',
                        help='Count data directory to evaluate counts from')
    args = parser.parse_args()
    main(args)

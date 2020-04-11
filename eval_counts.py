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
    kl_divergence, set_counts
from hab_ml.utils.logger import Logger

COUNTS_CSV = 'master_counts_v4.csv'


def main(args):
    # Excluded 3 classes atm
    classes = ['Akashiwo', 'Ceratium falcatiforme or fusus', 'Ceratium furca',
               'Cochlodinium', 'Lingulodinium polyedra', 'Prorocentrum micans']

    # Initialize logging
    input_dir = args.input_dir
    log_fname = os.path.join(input_dir, 'eval_counts.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')
    logger = logging.getLogger('create-csv')

    COUNT = 'cells/mL'
    logger.info('Count form: {}'.format(COUNT))

    # Set count forms
    MICRO_ML, LAB_ML, PIER_ML = set_counts('gtruth', 'cells/mL')
    _, LAB_P_ML, PIER_P_ML = set_counts('predicted', 'cells/mL')
    MICRO_RC, LAB_RC, PIER_RC = set_counts('gtruth', 'raw count')
    MICRO, LAB_P_RC, PIER_P_RC = set_counts('predicted', 'raw count')
    _, LAB_NRC, PIER_NRC = set_counts('gtruth', 'nrmlzd raw count')
    _, LAB_P_NRC, PIER_P_NRC = set_counts('predicted', 'nrmlzd raw count')

    # Set count form comparisons for evaluation
    settings = [
        (MICRO_ML, LAB_ML),
        (MICRO_ML, PIER_ML),
        (LAB_ML, PIER_ML),

        (MICRO_ML, LAB_RC),
        (MICRO_ML, PIER_RC),
        (LAB_RC, PIER_RC),

        (MICRO_ML, LAB_NRC),
        (MICRO_ML, PIER_NRC),
        (LAB_NRC, PIER_NRC),

        (MICRO_RC, LAB_RC),
        # (MICRO_RC, LAB_P_RC),
        (MICRO_RC, PIER_RC),
        # (MICRO_RC, PIER_P_RC),
        (LAB_RC, PIER_RC),
        # (LAB_RC, LAB_P_RC),
        # (PIER_RC, PIER_P_RC)
    ]

    # Evalute counts
    scores = {}
    for (smpl_gold, smpl_test) in settings:
        Logger.section_break(f'{smpl_gold} vs {smpl_test}')
        scores[(smpl_gold, smpl_test)] = evaluate_counts(input_dir,
                                                         gtruth_smpl_mthd=smpl_gold,
                                                         exp_smpl_mthd=smpl_test,
                                                         classes=classes)
    print_eval(scores)

def evaluate_counts(input_dir, gtruth_smpl_mthd, exp_smpl_mthd, classes):
    """ Evaluates a setting and save it for plotting

    Args:
        input_dir (str): Parent directory to read counts from
        sample_method_gold_std (str): Gold standard column name
        sample_method_to_test (str): Tested column name
        classes (list): Classes

    """
    # Initialize logging
    suffix = f'{gtruth_smpl_mthd.replace(" ", "-")}_{exp_smpl_mthd.replace(" ", "-")}'
    if 'cells/mL' in suffix:
        suffix = suffix.replace('-cells/mL', '-cells')
    logger = logging.getLogger(__name__)

    # Load dataset
    data = pd.read_csv(os.path.join(input_dir, COUNTS_CSV))
    df = data.copy()
    df = df.dropna()

    logger.info('Dataset size: {}'.format(df.shape[0]))
    logger.info('Total dates: {}'.format(df['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(df['class'].nunique()))

    # transform dataset

    # evaluation
    eval_metrics = evaluate_classes(df, gtruth_smpl_mthd,
                                    exp_smpl_mthd,
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

    logger.info('\nSMAPE Results\n{}'.format('-' * 30))
    logger.info(eval_df['smape'].describe())

    logger.info('\nKLDIV Results\n{}'.format('-' * 30))
    logger.info(eval_df['kl'].describe())

    csv_fname = os.path.join(input_dir, 'eval_{}.csv'.format(suffix))
    eval_df.to_csv(csv_fname, index=False)
    logger.info(f'Saved eval csv as {csv_fname}')

    return results['smape'], results['kl']

def evaluate_classes(df, sample_method_gold_std_col, sample_method_to_test_col,
                     classes, overall=False, average=False):
    """ Evaluate over all classes

    Args:
        df: Dataset
        sample_method_gold_std_col (str): Column name to compare against (gtruth)
        sample_method_to_test_col (str): Column name to test
        classes(list): Classes to evaluate
        overall (bool):  Flag to compute aggregated score. Default False
        average (bool): Flag to compute average of all class scores. Default False

    Returns:
        dict: Evaluation metrics

    """
    logger = logging.getLogger(__name__)
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
    for setting, score in scores.items():
        logger.info('{:50} SMAPE:{:0.2f}'.format(str(setting), score[0]))

    Logger.section_break('Overall KLDIV Scores')
    for setting, score in scores.items():
        logger.info('{:50} KLDIV:{:0.2f}'.format(str(setting), score[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Counts')
    parser.add_argument('--input_dir', type=str,
                        default='/data6/phytoplankton-db/counts/',
                        help='Count data directory to evaluate counts from')
    args = parser.parse_args()
    main(args)

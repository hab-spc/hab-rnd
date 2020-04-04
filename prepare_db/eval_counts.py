import logging
import os
import sys
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


def evaluate(df, sample_method_gold_std_col, sample_method_to_test_col, classes):
    eval_metrics = {'class': [], 'mae': [], 'mse': [], 'smape':[], 'ccc':[],
                    'bray curtis':[], 'pearson':[], 'kl':[]}
    epsilon = 0.00001

    logger.info('Classes selected: {}'.format(classes))
    df = df[df['class'].isin(classes)].reset_index(drop=True)
    for cls in classes:
        # get data
        temp = df.loc[df['class'] == cls]
        smpl_gold_std, smpl_to_test = temp[sample_method_gold_std_col], temp[sample_method_to_test_col]


        eval_metrics['class'].append(cls)
        # MAE
        eval_metrics['mae'].append(mean_absolute_error(smpl_gold_std, smpl_to_test))
        # MSE
        eval_metrics['mse'].append(mean_squared_error(smpl_gold_std, smpl_to_test))
        # smape
        eval_metrics['smape'].append(smape(smpl_gold_std, smpl_to_test))
        # CCC
        eval_metrics['ccc'].append(concordance_correlation_coefficient(smpl_gold_std, smpl_to_test))
        # bray curtis
        eval_metrics['bray curtis'].append(distance.braycurtis(smpl_gold_std, smpl_to_test))
        # Pearson correlation coefficient
        eval_metrics['pearson'].append(stats.pearsonr(smpl_gold_std, smpl_to_test)[0])
        # KL Divergence
        smpl_gold_std = smpl_gold_std.apply(lambda x: epsilon if x == 0 else x)
        smpl_to_test = smpl_to_test.apply(lambda x: epsilon if x == 0 else x)

        # smpl_gold_std = smpl_gold_std[smpl_gold_std != 0]
        # smpl_to_test = smpl_to_test[smpl_to_test != 0]

        # smpl_gold_std = smpl_gold_std / np.sum(smpl_gold_std)
        # smpl_to_test = smpl_to_test / np.sum(smpl_to_test)

        eval_metrics['kl'].append(kl_divergence(smpl_gold_std, smpl_to_test))

    print(
        'Standard deviation (smape per class): {}'.format(np.std(eval_metrics['smape'])))

    eval_metrics['class'].append('average')
    eval_metrics['mae'].append(np.mean(eval_metrics['mae']))
    eval_metrics['mse'].append(np.mean(eval_metrics['mse']))
    eval_metrics['smape'].append(np.mean(eval_metrics['smape']))
    eval_metrics['ccc'].append(np.mean(eval_metrics['ccc']))
    eval_metrics['bray curtis'].append(np.mean(eval_metrics['bray curtis']))
    eval_metrics['pearson'].append(np.mean(eval_metrics['pearson']))
    eval_metrics['kl'].append(np.mean(eval_metrics['kl']))

    eval_metrics['class'].append('overall')
    eval_metrics['mae'].append(mean_absolute_error(df[sample_method_gold_std_col],
                                                   df[sample_method_to_test_col]))
    eval_metrics['mse'].append(mean_squared_error(df[sample_method_gold_std_col],
                                                  df[sample_method_to_test_col]))

    std = np.std(np.abs(df[sample_method_gold_std_col] - df[sample_method_to_test_col]))
    print('Standard deviation (abs diff): {}'.format(std))

    eval_metrics['smape'].append(smape(df[sample_method_gold_std_col],
                                       df[sample_method_to_test_col]))
    eval_metrics['ccc'].append(
        concordance_correlation_coefficient(df[sample_method_gold_std_col],
                                            df[sample_method_to_test_col]))
    eval_metrics['bray curtis'].append(
        distance.braycurtis(df[sample_method_gold_std_col],
                            df[sample_method_to_test_col]))
    eval_metrics['pearson'].append(stats.pearsonr(df[sample_method_gold_std_col],
                                                  df[sample_method_to_test_col])[0])

    eval_metrics['kl'].append(
        kl_divergence(df[sample_method_gold_std_col], df[sample_method_to_test_col]))

    return eval_metrics


def evaluate_counts(input_dir, sample_method_gold_std, sample_method_to_test,
                    classes, count='raw count'):
    # Initialize logging
    suffix = f'{sample_method_gold_std}-{sample_method_to_test}'
    logger = logging.getLogger(__name__)

    label_type = 'gtruth'
    sample_method_gold_std_col = f'{sample_method_gold_std} {count}'
    sample_method_to_test_col = f'{sample_method_to_test} {label_type} {count}'

    if sample_method_gold_std == 'lab':
        sample_method_gold_std_col = f'{sample_method_gold_std} {label_type} {count}'

    data = pd.read_csv(os.path.join(input_dir, COUNTS_CSV))
    df = data.copy()
    df = df.dropna()

    logger.info('Dataset size: {}'.format(df.shape[0]))
    logger.info('Total dates: {}'.format(df['datetime'].nunique()))
    logger.info('Total classes: {}\n'.format(df['class'].nunique()))

    # transform dataset

    # evaluation
    eval_metrics = evaluate(df, sample_method_gold_std_col, sample_method_to_test_col,
                            classes)

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
        1: ['Pseudo-nitzschia chain'],
        2: ['Pseudo-nitzschia chain', 'Prorocentrum micans'],
        3: ['Akashiwo', 'Ceratium furca', 'Prorocentrum micans'],
        4: ['Chattonella', 'Gyrodinium', 'Ceratium falcatiforme or fusus',
            'Ceratium furca'],
        5: ['Ceratium falcatiforme or fusus', 'Akashiwo', 'Lingulodinium polyedra',
            'Ceratium furca', 'Prorocentrum micans'],
        6: ['Ceratium falcatiforme or fusus', 'Ceratium furca', 'Cochlodinium',
            'Pseudo-nitzschia chain', 'Chattonella', 'Lingulodinium polyedra'],
        7: ['Akashiwo', 'Prorocentrum micans', 'Lingulodinium polyedra',
            'Pseudo-nitzschia chain', 'Chattonella', 'Gyrodinium', 'Cochlodinium'],
        8: ['Akashiwo', 'Chattonella', 'Cochlodinium', 'Gyrodinium',
            'Lingulodinium polyedra', 'Pseudo-nitzschia chain', 'Prorocentrum micans',
            'Ceratium furca'],
        9: ['Ceratium furca', 'Pseudo-nitzschia chain', 'Gyrodinium',
            'Ceratium falcatiforme or fusus', 'Prorocentrum micans', 'Akashiwo',
            'Cochlodinium', 'Lingulodinium polyedra', 'Chattonella'],
    }

    input_dir = '/data6/phytoplankton-db/counts/'
    log_fname = os.path.join(input_dir, 'eval_counts.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')
    logger = logging.getLogger('create-csv')

    SMAPE_VS_CLASS_EXP = False
    COUNT = 'raw count'
    logger.info('Count form: {}'.format(COUNT))

    if SMAPE_VS_CLASS_EXP:
        micro_lab_scores = []
        micro_pier_scores = []
        lab_pier_scores = []

        ml_kl_scores, mp_kl_scores, lp_kl_scores = [], [], []

        for idx, clses in RANDOM_SMPLED_CLSSES.items():
            logger.info('Total classes:{}'.format(idx))

            Logger.section_break('micro vs lab')
            ml_smape, ml_kl = evaluate_counts(input_dir,
                                              sample_method_gold_std='micro',
                                              sample_method_to_test='lab', classes=clses)
            micro_lab_scores.append(ml_smape)
            ml_kl_scores.append(ml_kl)

            Logger.section_break('micro vs pier')
            mp_smape, mp_kl = evaluate_counts(input_dir,
                                              sample_method_gold_std='micro',
                                              sample_method_to_test='pier',
                                              classes=clses)
            micro_pier_scores.append(mp_smape)
            mp_kl_scores.append(mp_kl)

            Logger.section_break('lab vs pier')
            lp_smape, lp_kl = evaluate_counts(input_dir,
                                              sample_method_gold_std='lab',
                                              sample_method_to_test='pier',
                                              classes=clses)
            lab_pier_scores.append(lp_smape)
            lp_kl_scores.append(lp_kl)

        logger.info(micro_lab_scores)
        logger.info(micro_pier_scores)
        logger.info(lab_pier_scores)

        logger.info(ml_kl_scores)
        logger.info(mp_kl_scores)
        logger.info(lp_kl_scores)

    else:
        Logger.section_break('micro vs lab')
        ml_smape = evaluate_counts(input_dir,
                                   sample_method_gold_std='micro',
                                   sample_method_to_test='lab', classes=classes,
                                   count=COUNT)

        Logger.section_break('micro vs pier')
        mp_smape = evaluate_counts(input_dir,
                                   sample_method_gold_std='micro',
                                   sample_method_to_test='pier', classes=classes,
                                   count=COUNT)

        Logger.section_break('lab vs pier')
        lp_smape = evaluate_counts(input_dir,
                                   sample_method_gold_std='lab',
                                   sample_method_to_test='pier', classes=classes,
                                   count=COUNT)

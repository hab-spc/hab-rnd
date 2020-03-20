import argparse
from datetime import datetime
import glob
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
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from validate_exp.v_utils import concordance_correlation_coefficient, smape
from scipy.spatial import distance
from scipy import stats

# Project level imports
from hab_ml.data.label_encoder import HABLblEncoder
from hab_ml.utils.constants import Constants as CONST
from hab_ml.utils.logger import Logger

COUNTS_CSV = 'master_counts_v2.csv'

sample_method_gold_std = 'micro'
sample_method = sample_method_to_test =  'lab'
SUFFIX = '{}-{}'.format(sample_method_gold_std, sample_method_to_test)

def create_eval_csv(input_dir):
    # Initialize logging
    log_fname = os.path.join(input_dir, 'eval_csv_{}.log'.format(SUFFIX))
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')
    logger = logging.getLogger('create-csv')

    data = pd.read_csv(os.path.join(input_dir, COUNTS_CSV))
    df = data.copy()
    df = df.dropna()

    """loop over for each sample method (lab & pier) and concatenate it to the main_df"""
    label = 'gtruth'
    temp_gtruth = df[df['label'] == 'gtruth']
    temp_gtruth = temp_gtruth.rename({f'{sample_method_to_test} total abundance': f'{sample_method_to_test} {label} total abundance',
                        'raw count': f'{sample_method_to_test} {label} raw count',
                        f'{sample_method_to_test} relative abundance': f'{sample_method_to_test} {label} relative abundance'}, axis=1)
    temp_gtruth = temp_gtruth.drop('label', axis=1)

    label = 'predicted'
    temp_pred = df[df['label'] == 'prediction']
    temp_pred = temp_pred.rename({f'{sample_method_to_test} total abundance': f'{sample_method_to_test} {label} total abundance',
                        'raw count': f'{sample_method_to_test} {label} raw count',
                        f'{sample_method_to_test} relative abundance': f'{sample_method_to_test} {label} relative abundance'}, axis=1)
    temp_pred = temp_pred.drop('label', axis=1)

    concat = temp_pred.merge(temp_gtruth, on=['class', 'datetime',
                                              'micro raw count',
                                              'micro relative abundance',
                                              'micro total abundance'])
    if sample_method_gold_std == 'lab':
        concat = concat.rename({'lab relative abundance_x': 'lab relative abundance'},
                               axis=1)

    # evaluation
    # loop over each class, take the mean absolute error and ccc, then store them into a data structure
    label_type = 'gtruth'
    sample_method_gold_std_col = f'{sample_method_gold_std} relative abundance'
    sample_method_to_test_col = f'{sample_method_to_test} {label_type} relative abundance'

    eval_metrics = {'class': [], 'mae': [], 'mse': [], 'smape':[], 'ccc':[],
                    'bray curtis':[], 'pearson':[], 'kl':[]}
    # 'ccc'
    classes = sorted(concat['class'].unique())
    for cls in classes:
        # get data
        temp = concat.loc[concat['class'] == cls]
        smpl_gold_std, smpl_to_test = temp[sample_method_gold_std_col], temp[sample_method_to_test_col]

        smpl_to_test = smpl_to_test.apply(lambda x: 0.0000001 if x == 0 else x)

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
        eval_metrics['kl'].append(stats.entropy(smpl_gold_std, smpl_to_test, base=10.))

    eval_metrics['class'].append('overall')
    eval_metrics['mae'].append(mean_absolute_error(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['mse'].append(mean_squared_error(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['smape'].append(smape(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['ccc'].append(concordance_correlation_coefficient(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['bray curtis'].append(distance.braycurtis(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['pearson'].append(stats.pearsonr(concat[sample_method_gold_std_col],
                                                  concat[sample_method_to_test_col])[0])
    eval_metrics['kl'].append(stats.entropy(concat[sample_method_gold_std_col],
                                            concat[sample_method_to_test_col], base=10.))

    # Create dataframe and save to csv
    eval_df = pd.DataFrame(eval_metrics)
    csv_fname = os.path.join(input_dir, 'eval_{}.csv'.format(SUFFIX))
    eval_df.to_csv(csv_fname, index=False)
    logger.info(f'Saved eval csv as {csv_fname}')

if __name__ == '__main__':
    create_eval_csv('/data6/phytoplankton-db/counts')

"""Get counts


python get_counts.py --pier DB/csv/hab_in_situ_summer2019.csv --ouptut_dir counts
python get_counts.py --lab DB/csv/hab_in_vitro_summer2019.csv --ouptut_dir counts
python get_counts.py --micro DB/csv/hab_micro_summer2019.csv --ouptut_dir counts

python get_counts.py


"""
# Standard dist imports
import argparse
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

# Project level imports
from data.label_encoder import HABLblEncoder
from utils.constants import Constants as CONST
from utils.logger import Logger

HAB_ONLY = True

GT_ROOT_DIR = '/data6/phytoplankton-db'
# To update the model, change this directory
# MODEL_DIR = '/data6/lekevin/hab-master/hab_ml/experiments/resnet18_pretrained_c34_workshop2019_2'
MODEL_DIR = '/data6/yuanzhouyuan/hab/hab-ml/experiments/baseline_new_weighted_loss'
CV_MODEL_DIR = '/data6/phytoplankton-db/models'

## INPUT FILES
VALID_DATES = f'{GT_ROOT_DIR}/valid_collection_dates_master.txt'
SAMPLE_METHODS_CSV = {
    # 'lab': f'{GT_ROOT_DIR}/csv/hab_in_vitro_summer2019.csv',
    ## 'micro': f'{ROOT_DIR}/csv/hab_micro_2017_2019.csv',
    # 'micro': f'{GT_ROOT_DIR}/csv/hab_micro_summer2019.csv', # Prorocentrum micans included
    # 'pier': f'{GT_ROOT_DIR}/csv/hab_in_situ_summer2019.csv',

    # 'lab': f'{MODEL_DIR}/hab_in_vitro_summer2019-predictions.csv',
    # 'micro': f'{GT_ROOT_DIR}/csv/hab_micro_summer2019.csv',
    # Prorocentrum micans included
    # 'pier': f'{MODEL_DIR}/hab_in_situ_summer2019-predictions.csv',

    'lab': f'{CV_MODEL_DIR}/cv_hab_in_vitro_summer2019-predictions.csv',
    'pier': f'{CV_MODEL_DIR}/cv_hab_in_situ_summer2019-predictions.csv',
    'micro': f'{GT_ROOT_DIR}/csv/hab_micro_2017_2019.csv',
}

## OUTPUT FILES
# v9 --other
VERSION = 'v11'
COUNTS_CSV = {
    'plot_format': 'master_counts_{version}-plot.csv'.format(version=VERSION),
    'other_format': 'master_counts_{version}-other.csv'.format(version=VERSION),
    'master_format': 'master_counts_{version}.csv'.format(version=VERSION)
}


def main(args):
    output_dir = args.output_dir

    log_fname = os.path.join(output_dir, 'get_counts.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create COUNTS CSV')
    logger = logging.getLogger('create-csv')

    # Read valid dates file
    valid_dates = open(VALID_DATES, 'r').read().splitlines()

    counts_df = pd.DataFrame()
    for sample_method in SAMPLE_METHODS_CSV:
        # Check for file existance
        input_csv = SAMPLE_METHODS_CSV[sample_method]
        if not os.path.exists(input_csv):
            raise OSError(f'{input_csv} not found.')

        if sample_method != 'micro':
            smpl_counts = get_counts(input_csv=input_csv,
                                     output_dir=output_dir,
                                     sample_method=sample_method)
        else:
            smpl_counts = pd.read_csv(input_csv)

        # first sampling method (lab)
        if counts_df.empty:
            counts_df = counts_df.append(smpl_counts)
        else:
            # second sampling case micro
            if sample_method != 'pier':
                counts_df = counts_df.merge(smpl_counts, on=['datetime', 'class'],
                                            how='outer')
                counts_df = counts_df.rename({'label_x': 'label'}, axis=1)
                counts_df = counts_df.drop('label_y', axis=1)
            else:
                counts_df = counts_df.merge(smpl_counts, on=['datetime', 'class',
                                                             'label'])

    counts_df = counts_df[counts_df['datetime'].isin(valid_dates)]

    logger.info('Counts successfully concatenated')
    csv_fname = os.path.join(output_dir, COUNTS_CSV['plot_format'])
    logger.info(f'Saving -plot version as {csv_fname}')
    counts_df.to_csv(csv_fname, index=False)

    logger.info('\nReformatting counts for error/agreement...')
    counts_eval_df = transpose_labels(counts_df, sort=True)

    if not HAB_ONLY:
        csv_fname = os.path.join(output_dir, COUNTS_CSV['other_format'])
        logger.info(f'Saving -master -other version as {csv_fname}')
        counts_eval_df.to_csv(csv_fname, index=False)

        logger.info('\nFiltering "Other" out of dataset...')
        counts_eval_df = counts_eval_df[counts_eval_df['class'] != "Other"].reset_index(
            drop=True)

    csv_fname = os.path.join(output_dir, COUNTS_CSV['master_format'])
    logger.info(f'Saving -master version as {csv_fname}')
    counts_eval_df.to_csv(csv_fname, index=False)



def get_counts(input_csv=None, input_dir=None, output_dir=None, sample_method='micro', eval=False):
    """ Reformat predictions into a count csv file or conduct evaluation

    Args:
        input_csv (str): Abs path to the csv file to be reformatted
        output_dir (str): Abs path to the output directory.
        sample_method (str): Sampling method
        eval (bool): Flag to initiate evaluation

    Returns:

    """

    def pivot_counts_table(data, le, label_col='label', hab_only=HAB_ONLY):
        # filter for only hab species
        data['class'] = data[label_col].apply(le.hab_map)
        if hab_only:
            data = data[data['class'].isin(le.hab_classes[:-1])]
        # pivot the data
        df = pd.pivot_table(data, values=label_col, aggfunc='count', index=['class'])
        df['class'] = df.index
        df = df.rename({label_col: f'{sample_method} raw count'}, axis=1)
        # set classes not found to raw count of 0
        classes_not_found = list(
            set(le.hab_classes[:-1]).difference(df['class'].unique()))
        logger.debug('Classes not found: {}'.format(classes_not_found))
        if classes_not_found:
            for cls in classes_not_found:
                df = df.append({'class': cls, f'{sample_method} raw count': 0},
                               ignore_index=True)

        df['label'] = 'gtruth' if label_col == 'label' else 'prediction'
        # compute total abundance
        df['{} total abundance'.format(sample_method)] = sum(df[f'{sample_method} raw count'])
        # compute relative abundance
        df['{} relative abundance'.format(sample_method)] = df[f'{sample_method} raw count'] / df['{} total abundance'.format(sample_method)] * 100.0
        return df


    assert os.path.exists(input_csv), 'CSV file does not exist'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Initialize logging
    logger = logging.getLogger(__name__)

    # Read in data
    le = HABLblEncoder(classes_fname='/data6/lekevin/hab-master/hab_ml/experiments/resnet18_pretrained_c34_workshop2019_2/train_data.info')
    logger.info('Sampling method: {}'.format(sample_method))
    logger.info('Data loaded from {}'.format(input_csv))

    # TODO take out the read csv and take in the actual dataframe ???
    df = pd.read_csv(input_csv)

    # reformat dataframe
    main_df = pd.DataFrame()
    for date, date_df in df.groupby('image_date'):
        logger.debug(f'[date: {date}]Reformating dataframes for gtruth')
        label_df = pivot_counts_table(data=date_df, le=le, label_col=CONST.LBL)

        if CONST.HAB_PRED in date_df.columns:
            logger.debug(f'[date: {date}]Reformating dataframes for predictions')
            pred_df = pivot_counts_table(data=date_df, le=le, label_col=CONST.HAB_PRED)
            label_df = label_df.append(pred_df, sort=False)

        label_df['datetime'] = date
        main_df = main_df.append(label_df, sort=False)
        logger.debug('Appended {} rows to main dataframe'.format(label_df.shape[0]))

    # Get cells/mL
    main_df = normalize_imaged_volume(main_df, sample_method)

    main_df = normalize_raw_count(main_df, sample_method)

    col_order = list(main_df.columns)
    main_df = main_df[[col_order[-1]] + col_order[:-1]]
    csv_fname = os.path.join(output_dir, f'{sample_method}.csv')
    logger.info(f'Saving data as {csv_fname}')
    main_df.to_csv(csv_fname, index=False)
    return main_df


def reformat_counts(sample_method, data):
    le = HABLblEncoder(
        classes_fname='/data6/lekevin/hab-master/hab_ml/experiments/resnet18_pretrained_c34_workshop2019_2/train_data.info')

    label_df = pivot_counts_table(sample_method, data=data, le=le, label_col=CONST.LBL)
    if CONST.HAB_PRED in data.columns:
        pred_df = pivot_counts_table(sample_method, data=data, le=le,
                                     label_col=CONST.HAB_PRED)
        label_df = label_df.append(pred_df, sort=False)

    return label_df


def pivot_counts_table(sample_method, data, le, label_col='label'):
    logger = logging.getLogger(__name__)
    # filter for only hab species
    data['class'] = data[label_col].apply(le.hab_map)
    data = data[data['class'].isin(le.hab_classes[:-1])]
    # pivot the data
    df = pd.pivot_table(data, values=label_col, aggfunc='count', index=['class'])
    df['class'] = df.index
    df = df.rename({label_col: f'{sample_method} raw count'}, axis=1)
    # set classes not found to raw count of 0
    classes_not_found = list(set(le.hab_classes[:-1]).difference(df['class'].unique()))
    logger.debug('Classes not found: {}'.format(classes_not_found))
    if classes_not_found:
        for cls in classes_not_found:
            df = df.append({'class': cls, f'{sample_method} raw count': 0},
                           ignore_index=True)

    df['label'] = 'gtruth' if label_col == 'label' else 'prediction'
    # compute total abundance
    df['{} total abundance'.format(sample_method)] = sum(
        df[f'{sample_method} raw count'])
    # compute relative abundance
    df['{} relative abundance'.format(sample_method)] = df[
                                                            f'{sample_method} raw count'] / \
                                                        df['{} total abundance'.format(
                                                            sample_method)] * 100.0
    return df


def transpose_labels(df, sort=False):
    """loop over for each sample method (lab & pier) and concatenate it to the main_df"""
    label = 'gtruth'
    temp_gtruth = df[df['label'] == 'gtruth']
    for sample_method_to_test in ['lab', 'pier']:
        temp_gtruth = temp_gtruth \
            .rename({
            f'{sample_method_to_test} total abundance': f'{sample_method_to_test} {label} total abundance',
            f'{sample_method_to_test} raw count': f'{sample_method_to_test} {label} raw count',
            f'{sample_method_to_test} nrmlzd raw count': f'{sample_method_to_test} {label} nrmlzd raw count',
            f'{sample_method_to_test} relative abundance': f'{sample_method_to_test} {label} relative abundance',
            f'{sample_method_to_test} cells/mL': f'{sample_method_to_test} {label} cells/mL'},
            axis=1)

    temp_gtruth = temp_gtruth.drop('label', axis=1)

    label = 'predicted'
    temp_pred = df[df['label'] == 'prediction']
    for sample_method_to_test in ['lab', 'pier']:
        temp_pred = temp_pred \
            .rename({
            f'{sample_method_to_test} total abundance': f'{sample_method_to_test} {label} total abundance',
            f'{sample_method_to_test} raw count': f'{sample_method_to_test} {label} raw count',
            f'{sample_method_to_test} nrmlzd raw count': f'{sample_method_to_test} {label} nrmlzd raw count',
            f'{sample_method_to_test} relative abundance': f'{sample_method_to_test} {label} relative abundance',
            f'{sample_method_to_test} cells/mL': f'{sample_method_to_test} {label} cells/mL'},
            axis=1)
    temp_pred = temp_pred.drop('label', axis=1)

    merge_col = ['class', 'datetime', 'sampling time']
    micro_col = [col for col in df.columns if col.startswith('micro')]
    if all(mc in df.columns for mc in micro_col):
        merge_col += micro_col

    concat = temp_pred.merge(temp_gtruth, on=merge_col)

    # sort dataframe
    if sort:
        col = sorted(concat.columns)
        concat = concat[col[:2] + [col[-1]] + col[2:-1]]

    return concat


def normalize_imaged_volume(data, sample_method):
    if sample_method == 'pier':
        normalization_factor = 160
    else:
        normalization_factor = 60
    data[f'{sample_method} cells/mL'] = data[
                                            f'{sample_method} raw count'] / normalization_factor
    return data

def normalize_raw_count(data, sample_method):
    normalization_factor = 1
    if sample_method == 'pier':
        normalization_factor = 160 / 60
    data[f'{sample_method} nrmlzd raw count'] = data[
                                                    f'{sample_method} raw count'] / normalization_factor
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get counts dataset')
    parser.add_argument('--output_dir', type=str,
                        default=f'{GT_ROOT_DIR}/counts',
                        help='Output directory to save counts data')
    args = parser.parse_args()
    main(args)

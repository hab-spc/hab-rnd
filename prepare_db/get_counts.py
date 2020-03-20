"""Get counts


python get_counts.py --pier DB/csv/hab_in_situ_summer2019.csv --ouptut_dir counts
python get_counts.py --lab DB/csv/hab_in_vitro_summer2019.csv --ouptut_dir counts
python get_counts.py --micro DB/csv/hab_micro_summer2019.csv --ouptut_dir counts

python get_counts.py


"""
# Standard dist imports
import argparse
from datetime import datetime
import glob
import logging
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve()
print(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, PROJECT_DIR.parents[0])
sys.path.insert(0, str(PROJECT_DIR.parents[1]) + '/hab_ml')
sys.path.insert(0, PROJECT_DIR.parents[1])
sys.path.insert(0, PROJECT_DIR.parents[2])

# Third party imports
import pandas as pd

# Project level imports
from hab_ml.data.label_encoder import HABLblEncoder
from hab_ml.utils.constants import Constants as CONST
from hab_ml.utils.logger import Logger

ROOT_DIR = '/data6/phytoplankton-db'
SAMPLE_METHODS_CSV = {
	'lab': f'{ROOT_DIR}/csv/hab_in_vitro_summer2019.csv',
	# 'micro': f'{ROOT_DIR}/csv/hab_micro_2017_2019.csv',
    'micro': f'{ROOT_DIR}/csv/hab_micro_summer2019.csv', # Prorocentrum micans included
    'pier': f'{ROOT_DIR}/csv/hab_in_situ_summer2019.csv',
}
COUNTS_CSV = 'master_counts_{version}.csv'.format(version='v2')
VALID_DATES = f'{ROOT_DIR}/valid_collection_dates_master.txt'
OUTPUT_DIR = f'{ROOT_DIR}/counts'

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    valid_dates = open(VALID_DATES, 'r').read().splitlines()

    counts_df = pd.DataFrame()
    for sample_method in SAMPLE_METHODS_CSV:
        # Check for file existance
        input_csv = SAMPLE_METHODS_CSV[sample_method]
        if not os.path.exists(input_csv):
            raise OSError(f'{input_csv} not found.')

        if sample_method != 'micro':
            smpl_counts = get_counts(input_csv=input_csv,
                                 output_dir=OUTPUT_DIR,
                                 sample_method=sample_method)
        else:
            smpl_counts = pd.read_csv(input_csv)

        if counts_df.empty:
            counts_df = counts_df.append(smpl_counts)
        else:
            counts_df = counts_df.merge(smpl_counts, on=['datetime', 'class'])
            counts_df = counts_df.rename({'label_x':'label'} ,axis=1)
            counts_df = counts_df.drop('label_y', axis=1)

    counts_df = counts_df[counts_df['datetime'].isin(valid_dates)]
    counts_df = counts_df[['datetime', 'class', 'label',
                           'micro total abundance', 'lab total abundance',
                           'pier total abundance',
                           'micro raw count', 'lab raw count',
                           'pier raw count',
                           'micro relative abundance', 'lab relative abundance',
                           'pier relative abundance'
                           ]]

    csv_fname = os.path.join(OUTPUT_DIR, COUNTS_CSV)
    logger.info(f'Concatenated counts for all sample methods saved as {csv_fname}')
    counts_df.to_csv(csv_fname, index=False)


def get_counts(input_csv=None, input_dir=None, output_dir=None, sample_method='micro', eval=False):
    """ Reformat predictions into a count csv file or conduct evaluation

    Args:
        input_csv (str): Abs path to the csv file to be reformatted
        output_dir (str): Abs path to the output directory.
        sample_method (str): Sampling method
        eval (bool): Flag to initiate evaluation

    Returns:

    """
    def pivot_counts_table(data, le, label_col='label'):
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
                df = df.append({'class': cls, f'{sample_method} raw count': 0}, ignore_index=True)

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
    log_fname = os.path.join(output_dir, '{}.log'.format(sample_method))
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create LAB CSV')
    logger = logging.getLogger('create-csv')

    # Read in data
    le = HABLblEncoder(classes_fname='/data6/lekevin/hab-master/hab_ml/experiments/resnet18_pretrained_c34_workshop2019_2/train_data.info')
    logger.info('Sampling method: {}'.format(sample_method))
    logger.info('Data loaded from {}'.format(input_csv))

    df = pd.read_csv(input_csv)

    # reformat dataframe
    main_df = pd.DataFrame()
    for date, date_df in df.groupby('image_date'):
        logger.debug(f'[date: {date}]Reformating dataframes for gtruth')
        label_df = pivot_counts_table(data=date_df, le=le, label_col=CONST.LBL)

        if CONST.PRED in date_df.columns:
            logger.debug(f'[date: {date}]Reformating dataframes for predictions')
            pred_df = pivot_counts_table(data=date_df, le=le, label_col=CONST.PRED)
            label_df = label_df.append(pred_df, sort=False)

        label_df['datetime'] = date
        main_df = main_df.append(label_df, sort=False)
        logger.debug('Appended {} rows to main dataframe'.format(label_df.shape[0]))

    col_order = list(main_df.columns)
    main_df = main_df[[col_order[-1]] + col_order[:-1]]
    csv_fname = log_fname.replace('.log', '.csv')
    logger.info(f'Saving data as {csv_fname}')
    main_df.to_csv(csv_fname, index=False)
    return main_df

if __name__ == '__main__':
    main()
"""Get pier data counts

#1 get microscopy data
#2 for each date get the time bins
#3 group by gtruth
#4 transpose and get the counts
#5 return this for multiple time windows

"""
import logging
import os
import warnings
from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from get_counts import transpose_labels, reformat_counts
from utils.logger import Logger
from validate_exp.v_utils import set_counts
from validate_exp.stat_fns import mase
from counts_analysis.c_utils import COUNTS_CSV, IMG_CSV, CLASSES, CORRELATED_CLASSES

FILTER_CLASSES_FLAG = True
CORRELATED_CLASSES_FLAG = True
CELLS_1000_FLAG = False

# Initialize logger
GT_ROOT_DIR = '/data6/phytoplankton-db'
log_fname = os.path.join(f'{GT_ROOT_DIR}/counts', 'get_tw_counts.log')
Logger(log_fname, logging.INFO, log2file=False)
Logger.section_break('Create COUNTS CSV')
logger = logging.getLogger('create-csv')


def normalize_imaged_volume(data, sample_method):
    normalization_factor = 60
    data[f'{sample_method} gtruth cells/mL'] = data[
                                                   f'{sample_method} gtruth raw count'] / normalization_factor
    return data


def normalize_raw_count(data, sample_method):
    normalization_factor = 1
    data[f'{sample_method} gtruth nrmlzd raw count'] = data[f'{sample_method} gtruth ' \
                                                            f'raw count'] / normalization_factor
    return data


def get_time_window_pier_data(datetime_dict, data, time_col='image_time'):
    """Get time windows within the pier dataset"""
    time_window_df = pd.DataFrame()
    data[time_col] = pd.to_datetime(data[time_col]).dt.strftime("%H:%M:%S")
    for date, date_df in data.groupby('image_date'):
        # Get time window
        start, end = str(datetime_dict.loc[date]['start_time']), \
                     str(datetime_dict.loc[date]['end_time'])

        mask = (date_df[time_col] >= start) & (date_df[time_col] <= end)
        df = date_df.loc[mask]
        df = reformat_counts('pier', df)
        df['datetime'] = date
        df['sampling time'] = str(datetime_dict.loc[date]['sampling time'])
        time_window_df = time_window_df.append(df)

    time_window_df = transpose_labels(time_window_df)

    time_window_df = normalize_imaged_volume(time_window_df, sample_method='pier')
    time_window_df = normalize_raw_count(time_window_df, sample_method='pier')
    return time_window_df

def set_time_windows(time_window, data, time_col='sampling time'):
    """Define the offsets of the time windows"""
    smpl_datetime = data[[time_col, 'datetime']]
    smpl_datetime[time_col] = pd.to_datetime(smpl_datetime[time_col], format="%H:%M")
    offset = pd.DateOffset(seconds=time_window)
    smpl_datetime['start_time'] = (smpl_datetime[time_col] - offset).dt.time
    smpl_datetime['end_time'] = (smpl_datetime[time_col] + offset).dt.time
    smpl_datetime[time_col] = smpl_datetime[time_col].dt.time
    smpl_datetime = smpl_datetime.drop_duplicates()
    smpl_datetime = smpl_datetime.set_index('datetime')
    return smpl_datetime


# ACTUAL CODE TO GET PLOT SCORES
data = pd.read_csv(COUNTS_CSV['counts'])


def filter_classes(df, classes):
    return df[df['class'].isin(classes)].reset_index(drop=True)


if FILTER_CLASSES_FLAG:
    logger.info('FILTER CLASSES: {}'.format(FILTER_CLASSES_FLAG))
    classes = CORRELATED_CLASSES if CORRELATED_CLASSES_FLAG else CLASSES
    logger.info('CORRELATED CLASSES: {}'.format(CORRELATED_CLASSES_FLAG))
    data = filter_classes(data, classes)

data_ = data[[col for col in data.columns if 'pier' not in col]]
raw_counts = set_counts('gtruth', 'raw count', micro_default=True)

pier_data = pd.read_csv(IMG_CSV['pier-pred'])
TIME_WINDOWS = [5, 10, 50, 100, 200, 500, 750, 1000, 1200, 1500, 1750, 2000]

# Run through micro, then lab
for i in raw_counts[:2]:
    plot_scores = defaultdict(list)
    for t_win in TIME_WINDOWS:
        print('Starting time window: {}'.format(t_win))
        smpl_datetime = set_time_windows(t_win / 2, data)

        data_t_win_pier = get_time_window_pier_data(smpl_datetime, pier_data)
        data_t_win = data_t_win_pier.merge(data_, on=['datetime', 'class'])

        if CELLS_1000_FLAG:
            # Save data
            fname = '/data6/phytoplankton-db/counts/master_counts-pier1000s.csv'
            data_t_win.to_csv(fname, index=False)


        # Save data
        # Calculate MASE over here (median over each class)
        def evaluate_stat_over_each_class(x, y, data, stat, grouping='class'):
            score = []
            for grp_name, grp_data in data.groupby(grouping):
                score.append(stat(grp_data[x], grp_data[y]))
            return np.median(score)


        score = evaluate_stat_over_each_class(x=i, y=raw_counts[2], data=data_t_win,
                                              stat=mase)
        plot_scores['mase'].append(score)
        plot_scores['time'].append(t_win)
    logger.info(plot_scores)

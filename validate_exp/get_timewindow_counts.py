"""Get pier data counts

#1 get microscopy data
#2 for each date get the time bins
#3 group by gtruth
#4 transpose and get the counts
#5 return this for multiple time windows

"""
import logging
import os
from collections import defaultdict

import pandas as pd

from get_counts import transpose_labels, reformat_counts
from utils.logger import Logger
from validate_exp.v_utils import smape, set_counts

GT_ROOT_DIR = '/data6/phytoplankton-db'
# To update the model, change this directory
MODEL_DIR = '/data6/lekevin/hab-master/hab_ml/experiments/resnet18_pretrained_c34_workshop2019_2'

## INPUT FILES
VALID_DATES = f'{GT_ROOT_DIR}/valid_collection_dates_master.txt'
SAMPLE_METHODS_CSV = {
    # 'lab': f'{GT_ROOT_DIR}/csv/hab_in_vitro_summer2019.csv',
    ## 'micro': f'{ROOT_DIR}/csv/hab_micro_2017_2019.csv',
    # 'micro': f'{GT_ROOT_DIR}/csv/hab_micro_summer2019.csv', # Prorocentrum micans included
    # 'pier': f'{GT_ROOT_DIR}/csv/hab_in_situ_summer2019.csv',

    'lab': f'{MODEL_DIR}/hab_in_vitro_summer2019-predictions.csv',
    'micro': f'{GT_ROOT_DIR}/csv/hab_micro_2017_2019.csv',
    # 'micro': f'{GT_ROOT_DIR}/csv/hab_micro_summer2019.csv',
    # Prorocentrum micans included
    'pier': f'{MODEL_DIR}/hab_in_situ_summer2019-predictions.csv',

    'counts': '/data6/phytoplankton-db/counts/master_counts_v6.csv',
}

# Classes
classes = ['Akashiwo',
           'Ceratium falcatiforme or fusus',
           'Ceratium furca',
           'Chattonella',
           'Cochlodinium',
           'Lingulodinium polyedra',
           'Prorocentrum micans']

TIME_WINDOWS = [5, 10, 50, 100, 200, 500, 750, 1000, 1200, 1500, 1750, 2000]

log_fname = os.path.join(f'{GT_ROOT_DIR}/counts', 'get_tw_counts.log')
Logger(log_fname, logging.INFO, log2file=False)
Logger.section_break('Create COUNTS CSV')
logger = logging.getLogger('create-csv')


def get_time_window_pier_data(datetime_dict, data, time_col='image_time'):
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
    return time_window_df


# ACTUAL CODE TO GET PLOT SCORES
data = pd.read_csv(SAMPLE_METHODS_CSV['counts'])
# data = data[data['class'].isin(classes)].reset_index(drop=True)
data_ = data[[col for col in data.columns if 'pier' not in col]]
raw_counts = set_counts('gtruth', 'raw count', micro_default=True)


def set_time_windows(time_window, data, time_col='sampling time'):
    smpl_datetime = data[[time_col, 'datetime']]
    smpl_datetime[time_col] = pd.to_datetime(smpl_datetime[time_col], format="%H:%M")
    offset = pd.DateOffset(seconds=time_window)
    smpl_datetime['start_time'] = (smpl_datetime[time_col] - offset).dt.time
    smpl_datetime['end_time'] = (smpl_datetime[time_col] + offset).dt.time
    smpl_datetime[time_col] = smpl_datetime[time_col].dt.time
    smpl_datetime = smpl_datetime.drop_duplicates()
    smpl_datetime = smpl_datetime.set_index('datetime')
    return smpl_datetime


pier_data = pd.read_csv(SAMPLE_METHODS_CSV['pier'])

for i in raw_counts[:2]:
    plot_scores = defaultdict(list)
    for t_win in TIME_WINDOWS:
        print('Starting time window: {}'.format(t_win))
        smpl_datetime = set_time_windows(t_win / 2, data)

        data_t_win_pier = get_time_window_pier_data(smpl_datetime, pier_data)
        data_t_win = data_t_win_pier.merge(data_, on=['datetime', 'class'])
        x, y = data_t_win[i], data_t_win[raw_counts[2]]

        # Save data
        plot_scores['smape'].append(smape(x, y))
        plot_scores['time'].append(t_win)
    logger.info(plot_scores)

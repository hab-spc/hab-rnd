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

    'counts': '/data6/phytoplankton-db/counts/master_counts_v7.csv',
}

# Classes
classes = ['Akashiwo',
           'Ceratium falcatiforme or fusus',
           'Ceratium furca',
           'Chattonella',
           'Cochlodinium',
           'Lingulodinium polyedra',
           'Prorocentrum micans']

TIME_WINDOWS = [5, 10, 50, 100, 200, 500, 750, 1000]

log_fname = os.path.join(f'{GT_ROOT_DIR}/counts', 'get_seq_tb_counts.log')
Logger(log_fname, logging.INFO, log2file=False)
Logger.section_break('Create COUNTS CSV')
logger = logging.getLogger('create-csv')


def get_time_bins_lab_data(datetime_dict, data, time_col='image_time'):
    time_window_df = pd.DataFrame()
    data[time_col] = pd.to_datetime(data[time_col]).dt.strftime("%H:%M:%S")
    for date, date_df in data.groupby('image_date'):
        # Get time window
        start, end = datetime_dict[date]['start'], datetime_dict[date]['end']

        mask = (date_df[time_col] >= start) & (date_df[time_col] <= end)
        df = date_df.loc[mask]
        df = reformat_counts('lab', df)
        df['datetime'] = date
        df['sampling time'] = datetime_dict[date]['start'] + '-' + datetime_dict[date][
            'end']
        time_window_df = time_window_df.append(df)

    time_window_df = transpose_labels(time_window_df)
    return time_window_df


# ACTUAL CODE TO GET PLOT SCORES
data = pd.read_csv(SAMPLE_METHODS_CSV['counts'])
# data = data[data['class'].isin(classes)].reset_index(drop=True)
data_ = data[[col for col in data.columns if 'lab' not in col]]
raw_counts = set_counts('gtruth', 'raw count', micro_default=True)


def set_time_bins(time_window, data, time_col='image_time'):
    data[time_col] = pd.to_datetime(data[time_col], format="%H:%M:%S")
    start_datetimes = data.groupby('image_date')['image_time'].min().to_dict()
    smpl_datetime = {}
    for date, start_time in start_datetimes.items():
        start_end = {}
        start_end['start'] = str(start_time).split()[1]
        start_end['end'] = str(start_time + pd.DateOffset(seconds=time_window)).split()[
            1]
        smpl_datetime[date] = start_end

    return smpl_datetime


lab_data = pd.read_csv(SAMPLE_METHODS_CSV['lab'])

x_col, y_col = raw_counts[0], raw_counts[1]
plot_scores = defaultdict(list)
for t_win in TIME_WINDOWS:
    print('Starting time bin: {}'.format(t_win))
    smpl_datetime = set_time_bins(t_win, lab_data)

    data_t_bin_lab = get_time_bins_lab_data(smpl_datetime, lab_data)
    data_t_bin = data_t_bin_lab.merge(data_, on=['datetime', 'class'])
    x, y = data_t_bin[x_col], data_t_bin[y_col]

    # Save data
    plot_scores['smape'].append(smape(x, y))
    plot_scores['time'].append(t_win)
logger.info(plot_scores)

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

TIME_WINDOW = 6


def transpose_labels(df, sample_method, sort=False):
    label = 'gtruth'
    temp_gtruth = df[df['label'] == 'gtruth']

    temp_gtruth = temp_gtruth \
        .rename({
        f'{sample_method} total abundance': f'{sample_method} {label} total abundance',
        f'{sample_method} raw count': f'{sample_method} {label} raw count',
        f'{sample_method} nrmlzd raw count': f'{sample_method} {label} nrmlzd raw count',
        f'{sample_method} relative abundance': f'{sample_method} {label} relative abundance',
        f'{sample_method} cells/mL': f'{sample_method} {label} cells/mL'},
        axis=1)
    temp_gtruth = temp_gtruth.drop('label', axis=1)

    label = 'predicted'
    temp_pred = df[df['label'] == 'prediction']
    temp_pred = temp_pred \
        .rename({
        f'{sample_method} total abundance': f'{sample_method} {label} total abundance',
        f'{sample_method} raw count': f'{sample_method} {label} raw count',
        f'{sample_method} nrmlzd raw count': f'{sample_method} {label} nrmlzd raw count',
        f'{sample_method} relative abundance': f'{sample_method} {label} relative abundance',
        f'{sample_method} cells/mL': f'{sample_method} {label} cells/mL'},
        axis=1)

    temp_pred = temp_pred.drop('label', axis=1)
    merge_col = ['class', 'datetime', 'sampling time']
    concat = temp_pred.merge(temp_gtruth, on=merge_col)
    return concat


def set_time_bins(data, days=0, hours=0, minutes=0, seconds=0, time_col='image_time', centered=False):
    data[time_col] = pd.to_datetime(data[time_col], format="%H:%M:%S")
    start_datetimes = data.groupby('image_date')['image_time'].min().to_dict()
    smpl_datetime = {}
    for date, start_time in start_datetimes.items():
        start_end = {}
        start_end['start'] = str(start_time).split()[1]
        start_end['end'] = str(start_time + pd.DateOffset(hours=0, minutes=0, seconds=0)).split()[1]
        smpl_datetime[date] = start_end

    return smpl_datetime

def get_counts(data, time_col='image_time', datetime_dict=None):
    time_window_df = pd.DataFrame()
    data[time_col] = pd.to_datetime(data[time_col]).dt.strftime("%H:%M:%S")
    for date, date_df in data.groupby('image_date'):
        print(date)
        if datetime_dict:
            # Get time window
            start, end = datetime_dict[date]['start'], datetime_dict[date]['end']

            mask = (date_df[time_col] >= start) & (date_df[time_col] <= end)
            date_df = date_df.loc[mask]

        camera = 'pier'
        df = reformat_counts(camera, date_df)
        df['datetime'] = date

        if datetime_dict:
            df['sampling time'] = datetime_dict[date]['start'] + '-' + datetime_dict[date][
                'end']
        else:
            df['sampling time'] = date_df[time_col].min() + '-' + date_df[time_col].max()
        time_window_df = time_window_df.append(df)

    time_window_df = transpose_labels(time_window_df, 'pier')
    return time_window_df

MAJORITY_CLASSIFIER = True
print('Majority Vote Classifier: {}'.format(MAJORITY_CLASSIFIER))
# csv_file = '/data6/lekevin/hab-master/phytoplankton-db/csv/hab_in_situ_inbetween-34min_summer2019.csv'
# csv_file = '/data6/yuanzhouyuan/hab/hab-ml/experiments/baseline_new_weighted_loss/hab_in_situ_inbetween_summer2019-predictions.csv'
# csv_file = '/data6/yuanzhouyuan/hab/hab-ml/experiments/baseline_new_weighted_loss/hab_in_situ_inbetween-34min_summer2019-predictions.csv'
## Majority vote model
if MAJORITY_CLASSIFIER:
    # csv_file = '/data6/yuanzhouyuan/hab/majority_vote_inbetween.csv'
    # csv_file = '/data6/yuanzhouyuan/hab/majority_vote_inbetween-34min.csv'
    # csv_file = '/data6/yuanzhouyuan/hab/majority_vote_hab_in_situ_summer2019_6hr-predictions.csv'
    csv_file = '/data6/yuanzhouyuan/hab/majority_vote_hab_in_situ_2017_2019-predictions.csv'
data = pd.read_csv(csv_file)

if MAJORITY_CLASSIFIER:
    data = data.drop('ml_hab_prediction', axis=1)
    data = data.rename({'vote': 'ml_hab_prediction'}, axis=1)

counts_df = get_counts(data)
# output_csv = '/data6/phytoplankton-db/counts/auto_pier_counts/inbetween_summer2019-raw_counts.csv'
# output_csv = '/data6/phytoplankton-db/counts/auto_pier_counts/inbetween_summer2019-34min_raw_counts.csv'

if MAJORITY_CLASSIFIER:
    # output_csv = '/data6/phytoplankton-db/counts/auto_pier_counts/inbetween_summer2019-majority-raw_counts.csv'
    # output_csv = '/data6/phytoplankton-db/counts/auto_pier_counts/inbetween_summer2019-majority-34min_raw_counts.csv'
    # output_csv = '/data6/phytoplankton-db/counts/auto_pier_counts/summer2019-majority-6hr_counts.csv'
    output_csv = '/data6/phytoplankton-db/counts/auto_pier_counts/2017_2019-majority-raw_counts.csv'
counts_df.to_csv(output_csv, index=False)
print('Dataset saved as {}'.format(output_csv))

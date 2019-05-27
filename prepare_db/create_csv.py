"""Create CSV file to hold image meta data

Main usages:
- [TEMP] Generate csv files of in situ images retrieved from the SPC database
- Generate csv files of in vitro images from the lab
- - - images given in year-month-day format, so need to process all images
at once

"""
# Standard dist imports
import glob
import logging
import os
import re
import sys
sys.path.insert(0, '/data6/lekevin/hab-spc/')

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
#TODO log dataset statistics from this
from utils.logger import Logger
from data.d_utils import train_val_split, preprocess_dataframe
from prepare_db.parse_csv import SPCParser

# Module level constants

def create_proro_csv(filter_word='proro', data_dir=None):
    """Create prorocentrum training/eval dataset ONLY for PRORO"""
    output_dir = os.path.join(data_dir, 'csv/proro')
    Logger(os.path.join(output_dir, 'proro_csv.log'), logging.DEBUG,
           log2file=False)
    Logger.section_break('Create Proro-CSV')
    logger = logging.getLogger('create-csv')

    proro_dir = [f for f in os.listdir(data_dir) if filter_word in f][::-1]

    proro_df = pd.DataFrame()

    for p_dir in proro_dir:
        data_dir_ = os.path.join(data_dir, p_dir)

        TRAINVAL = 'trainval' in data_dir_
        logger.info('Creating csv for {}'.format(p_dir))

        if TRAINVAL:
            # Prepare training and validation set
            proro_types = glob.glob(data_dir_ + '/*')

            # Parse images and labels from each annotation set into dataframe
            for p_type in proro_types:
                pd_dict = {}
                pd_dict['images'] = glob.glob(p_type + '/*')
                pd_dict['label'] = os.path.basename(p_type)
                proro_df = proro_df.append(pd.DataFrame(pd_dict))

            # Create image id
            proro_df['image_id'] = proro_df['images'].apply(
                SPCParser.create_img_id)
            
            # Save copy of raw data before preprocessing
            fn = 'proro_trainval_raw.csv'
            logger.debug('Saving raw version of dataset as {}'.format(fn))
            proro_df.to_csv(os.path.join(output_dir, fn))
            proro_df = preprocess_dataframe(proro_df, logger, proro=True,
                                            enable_uniform=True)

            # Train val split
            train_df, val_df = train_val_split(proro_df)

            # Save csv files
            train_fn = os.path.join(output_dir, 'proro_train.csv')
            val_fn = os.path.join(output_dir, 'proro_val.csv')
            train_df.to_csv(train_fn, index=False)
            val_df.to_csv(val_fn, index=False)
            logger.info('Saved as:\n\t{}\n\t{}\n'.format(train_fn, val_fn))

        else:
            # Prepare test set of unknown labels
            data = glob.glob(data_dir_ + '/*')
            logger.info('Total unlabeled images: {}'.format(len(data)))

            # Set up test dataframe
            pd_dict = {}
            pd_dict['images'] = data  # Parsed image filenames
            pd_dict['label'] = np.nan  # Unknown label
            proro_df = pd.DataFrame(pd_dict)
            test_fn = os.path.join(output_dir, 'proro_test.csv')
            proro_df.to_csv(test_fn, index=False)
            logger.info('Saved as:\n\t{}'.format(test_fn))


# create_proro_csv()

def create_density_csv(output_dir, micro_csv, image_csv,
                       log_fname='density_csv.log',
                       csv_fname='Density_data.csv',
                       gtruth_available=False):
    """ Create density estimate csv file for validation generation

    Args:
        output_dir (str): Absolute path to output directory
        micro_csv (str): Absolute path to microscopy csv file
        image_csv (str): Absolute path to spc image csv file

    Returns:
        None

    """
    Logger(os.path.join(output_dir, log_fname), logging.DEBUG,
           log2file=False)
    Logger.section_break('Create Density-CSV')
    logger = logging.getLogger('create-csv')

    # Load data
    micro_data = pd.read_csv(micro_csv)
    image_data = pd.read_csv(image_csv)

    # Filter Image_data into filtered day estimates
    time_col = 'image_timestamp'
    time_dist = ['1min', '5min', '15min', '30min', '45min', '1h30min']
    time_img_data = SPCParser.get_time_density(image_data, time_col=time_col,
                                               time_bin=time_dist[0])
    for t in time_dist[1:]:
        temp = SPCParser.get_time_density(image_data, time_col=time_col, time_bin=t)
        time_img_data = time_img_data.merge(temp, on=time_col)
    time_img_data[time_col] = time_img_data[time_col].astype(str)

    # Get cell counts from Image_Data
    if gtruth_available:
        time_img_data = SPCParser.get_gtruth_counts(time_img_data)

    # Process Microscopy_data
    CSV_COLUMNS = '{},Prorocentrum micans (Cells/L),Time Collected (PST)'.format(time_col)
    micro_data = micro_data.rename(columns={'Datemm/dd/yy': time_col}, index=str)
    micro_data[time_col] = pd.to_datetime(micro_data[time_col]).dt.strftime('%Y-%m-%d')
    micro_data = micro_data[CSV_COLUMNS.split(',')]

    # Merge two data types
    density_data = micro_data.merge(time_img_data, on=time_col)

    # Rename columns for simplicity
    rename_dict = {'Prorocentrum micans (Cells/L)': 'micro_proro'}
    density_data = density_data.rename(columns=rename_dict, index=str)

    # Save as raw data
    fname = os.path.join(output_dir, csv_fname)
    density_data.to_csv(fname, index=False)
    logger.info('CSV Completed. Saved to {}'.format(fname))

# ====================== begin: create density ======================= #
spc_v = '2017'
rel_dir_flag = False
root_dir = 'rawdata' if rel_dir_flag else '/data6/lekevin/hab-master/hab-rnd/rawdata'
print('Creating density data')
micro_csv = os.path.join(root_dir, "Micro-all_data.csv")
image_csv = os.path.join(root_dir, "SPC-all_data.csv".format(spc_v))
create_density_csv(output_dir=root_dir, micro_csv=micro_csv, image_csv=image_csv,
                   log_fname='Density17-18-all_data.log',
                   csv_fname='Density17-18-all_data.csv',
                   gtruth_available=False)

# ====================== end: create density ======================= #

def create_time_period_csv(output_csv, micro_csv=None):
    """Create time period csv for SPICI"""
    if micro_csv:
        df = pd.read_csv(micro_csv)

    # for each date, get the time, add 1 hour
    df['end_time'] = (pd.to_datetime(df['Time Collected (PST)'],
                                      format='%H%M') + pd.DateOffset(
        hours=1, minutes=30)).dt.time
    df['start_time'] = (pd.to_datetime(df['Time Collected (PST)'],
                                      format='%H%M') - pd.DateOffset(
        hours=1, minutes=30)).dt.time

    # reformat the date
    df['date'] = pd.to_datetime(df['Datemm/dd/yy']).dt.strftime('%Y-%m-%d')
    df['start_datetime'] = df['date'].astype(str) + ' ' + df[
        'start_time'].astype(str)
    df['end_datetime'] = df['date'].astype(str) + ' ' + df[
        'end_time'].astype(str)
    df['min_camera'] = 0.03
    df['max_camera'] = 0.07
    df['camera'] = 'SPCP2'
    column_order = ['start_datetime', 'end_datetime', 'min_camera',
                    'max_camera', 'camera']
    df = df[column_order]
    df.to_csv(output_csv, index=False, header=False)

# data_dir = '/data6/lekevin/hab-spc/rawdata/'
# micro_csv = os.path.join(data_dir, "Micro_data.csv")
# output_csv = os.path.join(data_dir, 'micro_time_period.txt')
# create_time_period_csv(output_csv=output_csv, micro_csv=micro_csv)

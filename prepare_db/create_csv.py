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

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
#TODO log dataset statistics from this
from prepare_db.logger import Logger
from prepare_db.parse_csv import SPCParser, get_time_density, get_lab_data

# Module level constants
ROOT_DIR = '/data6/phytoplankton-db'
LAB_IMG_DIR = f'{ROOT_DIR}/hab_in_vitro/images'
META_DATA = '{date}/001/00000_static_html'

def main():
    # create_proro_csv()

    """Example running update_csv"""
    # pred_json = os.path.join(source_dir, date, rel_path, 'gtruth.json')
    # csv_fname = os.path.join(source_dir, date, rel_path, 'meta.csv')
    # print(f'Updating {date}')
    # update_csv(csv_fname=csv_fname, pred_json=pred_json, save=True)

    # spc_v = '-all_data'
    # spc_v = '-20Class_17-18'
    # rel_dir_flag = False
    # root_dir = 'rawdata' if rel_dir_flag else '/data6/lekevin/hab-master/hab-rnd/experiments/exp_hab20_2017_2018'
    # print('Creating density data')
    # micro_csv = os.path.join(root_dir, "Micro{}.csv".format(spc_v))
    # image_csv = os.path.join(root_dir, "SPC{}.csv".format(spc_v))
    # lab_csv = None
    # density_log = 'Density{}.log'.format(spc_v)
    # density_fname = 'Density{}.csv'.format(spc_v)
    # create_density_csv(output_dir=root_dir, micro_csv=micro_csv, image_csv=image_csv,
    #                    lab_csv=lab_csv,
    #                    log_fname=density_log,
    #                    csv_fname=density_fname,
    #                    gtruth_available=False)
    csv_fname = os.path.join(ROOT_DIR, 'csv', 'invitro_summer2019.csv')
    create_lab_csv(image_dir=LAB_IMG_DIR, csv_fname=csv_fname, save=True)


def create_density_csv(output_dir, micro_csv=None, image_csv=None, lab_csv=None,
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
    time_dist = '200s'
    num_of_classes=24

    # Process Microscopy_data
    micro_data = micro_data.rename(columns={'Datemm/dd/yy': time_col}, index=str)
    #TODO include preprocessing down here for Liter --> ml
    micro_data[time_col] = pd.to_datetime(micro_data[time_col]).dt.strftime('%Y-%m-%d')

    # Process Pier Camera Data
    time_img_data = get_time_density(image_data, time_col=time_col,
                                     time_freq=time_dist, insitu=True,
                                     num_of_classes=num_of_classes,
                                     save_dir=output_dir)
    time_img_data[time_col] = time_img_data[time_col].astype(str)

    # Merge two data types
    density_data = micro_data.merge(time_img_data, on=time_col)

    # Get cell counts from Image_Data
    if gtruth_available:
        time_img_data = SPCParser.get_gtruth_counts(time_img_data)

    # Get lab time densities
    if lab_csv:
        lab_data = pd.read_csv(lab_csv)
        lab_data = lab_data.rename(columns={'timestamp': time_col}, index=str)
        time_dist = '200s'
        time_lab_data = SPCParser.get_time_density(lab_data, time_col=time_col,
                                                   time_bin=time_dist)
        time_lab_data[time_col] = time_lab_data[time_col].astype(str)
        density_data = density_data.merge(time_lab_data, on=time_col)

    # Rename columns for simplicity
    rename_dict = {'Prorocentrum micans (Cells/L)': 'micro_proro'}
    density_data = density_data.rename(columns=rename_dict, index=str)

    # Save as raw data
    fname = os.path.join(output_dir, csv_fname)
    density_data.to_csv(fname, index=False)
    logger.info('CSV Completed. Saved to {}'.format(fname))

def create_time_period_csv(output_csv, micro_csv=None, datefmt='%Y%m%d', timefmt='%H%M', offset_hours=0,
                           offset_min=0, min_camera=0.03, max_camera=0.07, camera='SPCP2'):
    """Create time period csv for SPICI"""
    if micro_csv:
        df = pd.read_csv(micro_csv)

    # for each date, get the time, add 1 hour
    time_col = 'Time Collected hhmm (PST)'
    df['end_time'] = (pd.to_datetime(df[time_col],
                                     format=timefmt) + pd.DateOffset(
        hours=offset_hours, minutes=offset_min)).dt.time
    df['start_time'] = (pd.to_datetime(df[time_col],
                                       format=timefmt) - pd.DateOffset(
        hours=offset_hours, minutes=offset_min)).dt.time

    # reformat the date
    date_col = 'SampleID (YYYYMMDD)'
    df['date'] = pd.to_datetime(df[date_col], format=datefmt).dt.strftime('%Y-%m-%d')
    df['start_datetime'] = df['date'].astype(str) + ' ' + df[
        'start_time'].astype(str)
    df['end_datetime'] = df['date'].astype(str) + ' ' + df[
        'end_time'].astype(str)
    df['min_camera'] = min_camera
    df['max_camera'] = max_camera
    df['camera'] = camera
    column_order = ['start_datetime', 'end_datetime', 'min_camera',
                    'max_camera', 'camera']
    df = df[column_order]
    print(f'Saved file to {output_csv}')
    df.to_csv(output_csv, index=False, header=False)

# data_dir = '/data6/lekevin/hab-spc/rawdata/'
# micro_csv = os.path.join(data_dir, "Micro_data.csv")
# output_csv = os.path.join(data_dir, 'micro_time_period.txt')
# create_time_period_csv(output_csv=output_csv, micro_csv=micro_csv)

def create_lab_csv(image_dir=None, csv_fname='', raw_color=False, save=False):
    log_fname = csv_fname.replace('.csv', '.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create LAB CSV')
    logger = logging.getLogger('create-csv')

    # first two dates and last date are scrapped
    sampled_dates = sorted(glob.glob(os.path.join(image_dir, '*')))[2:-1]
    lab_df = pd.DataFrame()
    bad_dates = []
    for date in sampled_dates:
        # merge the tsv and predictions.json
        # create image_id from the url
        try:
            meta_df = get_lab_data(date_dir=META_DATA.format(date=date))

            if raw_color:
                meta_df['images'] = meta_df['images'].apply(
                    lambda x: x.replace('.jpeg', 'raw_color.jpeg'))

            if save:
                meta_fname = os.path.join(META_DATA.format(date=date), 'meta.csv')
                meta_df.to_csv(meta_fname, index=False)
                logger.info(f'Saved as {meta_fname}')

            lab_df = lab_df.append(meta_df)
        except:
            logger.error(f'ERROR: Csv not generated for {date}')
            bad_dates.append(date)
            continue
    lab_df.to_csv(csv_fname, index=False)
    logger.info(f'Lab dataset generation completed.\nSaved as {csv_fname}')
    logger.info('{} Bad dates to review\n{}'.format(len(bad_dates), bad_dates))

def update_csv(csv_fname='meta.csv', pred_json='predictions.json', save=False):
    """ Update the csv file with new groundtruths

    Args:
        csv_fname: Abspath to  meta csv file to merge
        pred_json: Abspath to predictions json file to merge
        save: Flag to save csv file

    Returns:
        merged dataframe

    """
    import os
    import json
    import pandas as pd
    assert os.path.exists(csv_fname), print(csv_fname)
    assert os.path.exists(pred_json), print(pred_json)

    # Read in files and preprocess
    label_col = 'label'
    meta_df = pd.read_csv(csv_fname)
    pred = json.load(open(pred_json, 'rb'))

    # Drop outdated `label` column (used as gtruth in machine learning exp)
    meta_df = meta_df.drop(columns=label_col, axis=0)
    meta_df = meta_df.rename({'timestamp': 'image_timestamp'})

    # Preprocess prediction json
    pred_df = pd.DataFrame(pred['machine_labels'])
    pred_df = pred_df.rename({'gtruth': label_col}, axis=1)

    # Merge based off image_id
    merged = meta_df.merge(pred_df, on='image_id')

    # Overwrite previous csv file with new gtruth
    if save:
        merged.to_csv(csv_fname, index=False)
    return merged

if __name__ == '__main__':
    main()

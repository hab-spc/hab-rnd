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
from pprint import pformat

# Third party imports
import pandas as pd

# Project level imports
# TODO log dataset statistics from this
from prepare_db.logger import Logger
from prepare_db.parse_csv import SPCParser, get_time_density, get_lab_data

# Module level constants
ROOT_DIR = '/data6/phytoplankton-db'
LAB_IMG_DIR = f'{ROOT_DIR}/hab_in_vitro/images'
META_DATA = '{date}/001/00000_static_html'

"""WHEN USING NEW MODEL, UPDATE THIS"""
CLASSES_INFO = '/data6/lekevin/hab-master/hab_rnd/hab-ml/experiments/hab_model_v1:20191023/train_data.info'

def main():
    i = 0
    # spc_v = '-all_data' # outdated
    # spc_v = '-20Class_17-18' # outdated
    """Uncomment to create a density csv file to do sampling correlations"""
    spc_v = '-20Class_2017_2019'
    spc_v = '-20Class_summer2019'

    if 'summer' in spc_v:
        root_dir = '/data6/lekevin/hab-master/hab_rnd/experiments/exp_hab20_summer2019'
    elif '2017_2019' in spc_v:
        root_dir = '/data6/lekevin/hab-master/hab_rnd/experiments/exp_hab20_2017_2019'
    else:
        root_dir = 'rawdata'

    print('Creating density data')
    micro_csv = os.path.join(root_dir, "Micro{}.csv".format(spc_v))
    insitu_csv = os.path.join(root_dir, "Insitu{}.csv".format(spc_v))
    invitro_csv = os.path.join(root_dir, "Invitro{}.csv".format(spc_v))
    density_log = 'Density{}.log'.format(spc_v)
    density_fname = 'Density{}.csv'.format(spc_v)
    """don't think output_dir is necessary, just the three csv filenames + density"""
    create_density_csv(output_dir=root_dir, micro_csv=micro_csv, insitu_csv=insitu_csv,
                       invitro_csv=invitro_csv,
                       log_fname=density_log,
                       csv_fname=density_fname,
                       gtruth_available=False)

    """Uncomment to create a in vitro csv file containing all images, processed times, etc."""
    # csv_fname = os.path.join(ROOT_DIR, 'csv', 'hab_in_vitro_summer2019.csv')
    # create_lab_csv(image_dir=LAB_IMG_DIR, csv_fname=csv_fname, save=True)


def create_density_csv(output_dir, micro_csv=None, insitu_csv=None, invitro_csv=None,
                       log_fname='density_csv.log',
                       csv_fname='Density_data.csv',
                       gtruth_available=False):
    """ Create density estimate csv file for validation generation

    Args:
        output_dir (str): Absolute path to output directory
        micro_csv (str): Absolute path to microscopy csv file
        insitu_csv (str): Absolute path to spc image csv file

    Returns:
        None

    """
    Logger(os.path.join(output_dir, log_fname), logging.DEBUG,
           log2file=False)
    Logger.section_break('Create Density-CSV')
    logger = logging.getLogger('create-csv')

    micro_data = pd.read_csv(micro_csv)
    insitu_data = pd.read_csv(insitu_csv)

    # Initialize csv parsing instance for useful item grabs
    spc = SPCParser(csv_fname=insitu_csv, classes=CLASSES_INFO)
    # Filter the microscopy classes based on trained HAB classes from the insitu
    micro_data, trained_classes = spc.filter_classes(micro_data)

    # Filter Image_data into filtered day estimates
    time_col = 'image_timestamp'
    time_dist = '200s'

    # Process Microscopy_data
    micro_classes = list(micro_data.columns[3:])
    micro_data = micro_data.rename(columns={'SampleID (YYYYMMDD)': time_col}, index=str)
    micro_data[micro_classes] = micro_data[micro_classes].apply(lambda x: x / 1000, axis=1)
    micro_data[time_col] = pd.to_datetime(micro_data[time_col], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    # Process Pier Camera Data
    time_img_data = get_time_density(insitu_data, time_col=time_col,
                                     time_freq=time_dist, insitu=True,
                                     num_of_classes=len(spc.cls2idx),
                                     save_dir=output_dir)
    time_img_data[time_col] = time_img_data[time_col].astype(str)

    # Filter the classes for the insitu dataset after conversion
    clss = []
    for i in list(trained_classes.values()):
        clss.append('pier_{}_avg_{}'.format(i, time_dist))
        clss.append('pier_{}_std_{}'.format(i, time_dist))
    time_img_data = time_img_data[list(time_img_data.columns[:4]) + clss]

    # Merge two data types
    bad_dates = set(micro_data['image_timestamp']).difference(time_img_data['image_timestamp'])
    logger.debug('Dates camera under maintenance ({})\n{}\n{}'.format(len(bad_dates), '-' * 10, pformat(bad_dates)))
    density_data = micro_data.merge(time_img_data, on=time_col)

    # Get cell counts from Image_Data
    if gtruth_available:
        time_img_data = SPCParser.get_gtruth_counts(time_img_data)

    # Get lab time densities
    if invitro_csv:
        lab_data = pd.read_csv(invitro_csv)
        time_dist = '200s'
        time_lab_data = get_time_density(lab_data, time_col=time_col,
                                         time_freq=time_dist, insitu=False,
                                         num_of_classes=len(spc.cls2idx),
                                         save_dir=output_dir)
        time_lab_data[time_col] = time_lab_data[time_col].astype(str)
        density_data = density_data.merge(time_lab_data, on=time_col)

    # Save as raw data
    fname = os.path.join(output_dir, csv_fname)
    density_data.to_csv(fname, index=False)
    logger.info('CSV Completed. Saved to {}'.format(fname))
    logger.info('Trained classes\n{}\n{}'.format('-' * 10, pformat(trained_classes)))

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

def create_lab_csv(image_dir=None, csv_fname='', raw_color=False, save=False):
    """Create lab csv"""
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

if __name__ == '__main__':
    main()
